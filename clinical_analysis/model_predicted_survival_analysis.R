# =========================
# Clinical association analysis
# using threshold-based model-predicted LVI group
# KM curves + univariable Cox regression
# =========================

library(readr)
library(readxl)
library(dplyr)
library(survival)
library(survminer)
library(ggplot2)
library(tibble)
library(stringr)
library(optparse)

# -------------------------
# Helper: required CLI arguments
# -------------------------
check_required_args <- function(opt, args) {
  missing_args <- args[sapply(args, function(x) {
    is.null(opt[[x]]) || is.na(opt[[x]]) || opt[[x]] == ""
  })]

  if (length(missing_args) > 0) {
    stop("Missing required arguments: ", paste(missing_args, collapse = ", "))
  }
}

check_input_file <- function(path, label) {
  if (!file.exists(path)) {
    stop(label, " does not exist: ", path)
  }
}

# -------------------------
# 1. Command-line arguments
# -------------------------

option_list <- list(
  make_option(c("--pred-file"), dest = "pred_file", type = "character",
              help = "Prediction TSV file with patient_id, slide_id, y_true, and model_score."),
  make_option(c("--clin-file"), dest = "clin_file", type = "character",
              help = "Clinical Excel file with survival endpoints."),
  make_option(c("--out-dir"), dest = "out_dir", type = "character",
              help = "Output directory."),
  make_option(c("--selected-threshold"), dest = "selected_threshold", type = "double", default = NA_real_,
              help = "Optional decision threshold. If omitted, extracted from prediction filename."),
  make_option(c("--sheet"), dest = "sheet", type = "integer", default = 1,
              help = "Excel sheet index or name for the clinical file [default: %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))
check_required_args(opt, c("pred_file", "clin_file", "out_dir"))

pred_file <- opt$pred_file
clin_file <- opt$clin_file
out_dir <- opt$out_dir
selected_threshold <- opt$selected_threshold
sheet <- opt$sheet

check_input_file(pred_file, "Prediction file")
check_input_file(clin_file, "Clinical file")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2. Read data
# -------------------------
pred <- read_tsv(pred_file, show_col_types = FALSE)
clin <- read_excel(clin_file, sheet = sheet)

# -------------------------
# 3. Determine selected threshold
# -------------------------
# By default, the script extracts the threshold from filenames such as:
# external_predictions_selectedthr0p5865.tsv -> 0.5865
# Alternatively, pass --selected-threshold manually.

if (is.na(selected_threshold)) {
  filename <- basename(pred_file)
  thr_text <- str_match(filename, "selectedthr([0-9p]+)")[, 2]

  if (is.na(thr_text)) {
    stop("Could not extract selected threshold from filename. Please provide --selected-threshold manually.")
  }

  selected_threshold <- as.numeric(str_replace(thr_text, "p", "."))
}

cat("Selected threshold =", selected_threshold, "\n")

# -------------------------
# 4. Basic checks
# -------------------------
required_pred_cols <- c("patient_id", "slide_id", "y_true", "model_score")
required_clin_cols <- c(
  "patient_id",
  "fu_survival", "surv_outcome",
  "RFi_time", "RFi_event",
  "DRFi_time", "DRFi_event"
)

missing_pred <- setdiff(required_pred_cols, colnames(pred))
missing_clin <- setdiff(required_clin_cols, colnames(clin))

if (length(missing_pred) > 0) {
  stop(paste("Missing columns in prediction file:", paste(missing_pred, collapse = ", ")))
}
if (length(missing_clin) > 0) {
  stop(paste("Missing columns in clinical file:", paste(missing_clin, collapse = ", ")))
}

# -------------------------
# 5. Create threshold-based model-predicted LVI group
# -------------------------
pred_group <- pred %>%
  mutate(
    patient_id = as.character(patient_id),
    model_score = as.numeric(model_score),
    model_pred_LVI = ifelse(model_score >= selected_threshold, 1, 0),
    model_LVI_group = factor(
      ifelse(model_pred_LVI == 1, "Model-predicted LVI-positive", "Model-predicted LVI-negative"),
      levels = c("Model-predicted LVI-negative", "Model-predicted LVI-positive")
    )
  ) %>%
  select(patient_id, slide_id, y_true, model_score, model_pred_LVI, model_LVI_group)

# Check duplicate patient IDs
if (any(duplicated(pred_group$patient_id))) {
  stop("Duplicated patient_id found in prediction table. Please check patient-slide mapping.")
}

cat("\nModel-predicted LVI group counts:\n")
print(table(pred_group$model_LVI_group, useNA = "ifany"))

cat("\nGround-truth y_true vs model-predicted LVI group:\n")
print(table(pred_group$y_true, pred_group$model_LVI_group, useNA = "ifany"))

# -------------------------
# 6. Merge with clinical data
# -------------------------
dat <- clin %>%
  mutate(patient_id = as.character(patient_id)) %>%
  inner_join(pred_group, by = "patient_id") %>%
  mutate(
    fu_survival = as.numeric(fu_survival),
    surv_outcome = as.numeric(surv_outcome),
    RFi_time = as.numeric(RFi_time),
    RFi_event = as.numeric(RFi_event),
    DRFi_time = as.numeric(DRFi_time),
    DRFi_event = as.numeric(DRFi_event),
    
    # OS: death from any cause
    OS_time = fu_survival,
    OS_event = ifelse(surv_outcome %in% c(1, 2), 1, 0),
    
    # BCSS: breast cancer death only
    BCSS_time = fu_survival,
    BCSS_event = ifelse(surv_outcome == 1, 1, 0)
  )

cat("\nMerged patient number:", nrow(dat), "\n")

write_tsv(dat, file.path(out_dir, "merged_model_predicted_LVI_clinical_data.tsv"))

# -------------------------
# 7. Helper function: KM plot
# -------------------------
make_km_plot <- function(data, time_col, event_col, endpoint_name, out_dir) {
  
  df <- data %>%
    mutate(
      .time = as.numeric(.data[[time_col]]),
      .event = as.numeric(.data[[event_col]])
    ) %>%
    filter(
      !is.na(.time),
      !is.na(.event),
      !is.na(model_LVI_group)
    )
  
  cat("\n=========================\n")
  cat(endpoint_name, "\n")
  cat("=========================\n")
  cat("N =", nrow(df), "\n")
  cat("Events =", sum(df$.event == 1, na.rm = TRUE), "\n")
  print(table(df$model_LVI_group, useNA = "ifany"))
  print(table(df$model_LVI_group, df$.event, useNA = "ifany"))
  
  fit <- survfit(Surv(.time, .event) ~ model_LVI_group, data = df)
  
  p <- ggsurvplot(
    fit,
    data = df,
    risk.table = TRUE,
    pval = TRUE,
    conf.int = FALSE,
    xlab = "Time (years)",
    ylab = paste0(endpoint_name, " probability"),
    legend.title = "Model group",
    legend.labs = c("Model-predicted LVI-negative", "Model-predicted LVI-positive"),
    title = paste0(endpoint_name, " by model-predicted LVI group"),
    risk.table.height = 0.25,
    ggtheme = theme_bw()
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_model_predicted_LVI_plot.png")),
    plot = p$plot,
    width = 6,
    height = 5,
    dpi = 300
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_model_predicted_LVI_risk_table.png")),
    plot = p$table,
    width = 6,
    height = 2.5,
    dpi = 300
  )
  
  pdf(
    file.path(out_dir, paste0("KM_", endpoint_name, "_by_model_predicted_LVI_combined.pdf")),
    width = 7,
    height = 7
  )
  print(p)
  dev.off()
  
  return(fit)
}

# -------------------------
# 8. Helper function: univariable Cox regression
# -------------------------
run_cox <- function(data, time_col, event_col, endpoint_name) {
  
  df <- data %>%
    mutate(
      .time = as.numeric(.data[[time_col]]),
      .event = as.numeric(.data[[event_col]])
    ) %>%
    filter(
      !is.na(.time),
      !is.na(.event),
      !is.na(model_LVI_group)
    )
  
  cox_fit <- coxph(Surv(.time, .event) ~ model_LVI_group, data = df)
  s <- summary(cox_fit)
  
  result <- tibble(
    endpoint = endpoint_name,
    n = nrow(df),
    events = sum(df$.event == 1, na.rm = TRUE),
    comparison = "Model-predicted LVI-positive vs Model-predicted LVI-negative",
    HR = s$coefficients[1, "exp(coef)"],
    lower_95_CI = s$conf.int[1, "lower .95"],
    upper_95_CI = s$conf.int[1, "upper .95"],
    p_value = s$coefficients[1, "Pr(>|z|)"]
  )
  
  return(result)
}

# -------------------------
# 9. Kaplan-Meier curves
# -------------------------
fit_rfi <- make_km_plot(dat, "RFi_time", "RFi_event", "RFi", out_dir)
fit_drfi <- make_km_plot(dat, "DRFi_time", "DRFi_event", "DRFi", out_dir)
fit_os <- make_km_plot(dat, "OS_time", "OS_event", "OS", out_dir)
fit_bcss <- make_km_plot(dat, "BCSS_time", "BCSS_event", "BCSS", out_dir)

# -------------------------
# 10. Cox regression
# -------------------------
cox_results <- bind_rows(
  run_cox(dat, "RFi_time", "RFi_event", "RFi"),
  run_cox(dat, "DRFi_time", "DRFi_event", "DRFi"),
  run_cox(dat, "OS_time", "OS_event", "OS"),
  run_cox(dat, "BCSS_time", "BCSS_event", "BCSS")
)

print(cox_results)

write_tsv(
  cox_results,
  file.path(out_dir, "cox_univariable_model_predicted_LVI_results.tsv")
)

cat("\nDone. Outputs saved to:\n")
cat(out_dir, "\n")