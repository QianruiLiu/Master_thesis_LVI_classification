# =========================
# Clinical association of pathologist-assessed LVI
# using full clinical cohort
# KM curves + univariable Cox regression
# =========================

library(readxl)
library(dplyr)
library(readr)
library(survival)
library(survminer)
library(ggplot2)
library(tibble)
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
  make_option(c("--clin-file"), dest = "clin_file", type = "character",
              help = "Clinical Excel file with pathologist-assessed LVI and survival endpoints."),
  make_option(c("--out-dir"), dest = "out_dir", type = "character",
              help = "Output directory."),
  make_option(c("--sheet"), dest = "sheet", type = "integer", default = 1,
              help = "Excel sheet index or name for the clinical file [default: %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))
check_required_args(opt, c("clin_file", "out_dir"))

clin_file <- opt$clin_file
out_dir <- opt$out_dir
sheet <- opt$sheet

check_input_file(clin_file, "Clinical file")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2. Read clinical data
# -------------------------
clin <- read_excel(clin_file, sheet = sheet)

required_clin_cols <- c(
  "patient_id",
  "LVI",
  "fu_survival", "surv_outcome",
  "RFi_time", "RFi_event",
  "DRFi_time", "DRFi_event"
)

missing_clin <- setdiff(required_clin_cols, colnames(clin))
if (length(missing_clin) > 0) {
  stop(paste("Missing columns in clinical file:", paste(missing_clin, collapse = ", ")))
}

# -------------------------
# 3. Prepare analysis data
# -------------------------
dat <- clin %>%
  mutate(
    patient_id = as.character(patient_id),
    
    # Use full-cohort pathologist-assessed LVI label
    LVI_y_true = as.numeric(LVI),
    
    LVI_group = factor(
      ifelse(LVI_y_true == 1, "LVI-positive", "LVI-negative"),
      levels = c("LVI-negative", "LVI-positive")
    ),
    
    fu_survival = as.numeric(fu_survival),
    surv_outcome = as.numeric(surv_outcome),
    RFi_time = as.numeric(RFi_time),
    RFi_event = as.numeric(RFi_event),
    DRFi_time = as.numeric(DRFi_time),
    DRFi_event = as.numeric(DRFi_event),
    
    # OS: event = death from any cause
    OS_time = fu_survival,
    OS_event = ifelse(surv_outcome %in% c(1, 2), 1, 0),
    
    # BCSS: event = breast cancer death only
    BCSS_time = fu_survival,
    BCSS_event = ifelse(surv_outcome == 1, 1, 0)
  ) %>%
  filter(!is.na(LVI_y_true))

cat("\nFull cohort patient number:", nrow(dat), "\n")
cat("\nLVI group counts:\n")
print(table(dat$LVI_group, useNA = "ifany"))

write_tsv(dat, file.path(out_dir, "full_cohort_LVI_clinical_data.tsv"))

# -------------------------
# 4. Helper function: KM plot
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
      !is.na(LVI_group)
    )
  
  cat("\n=========================\n")
  cat(endpoint_name, "\n")
  cat("=========================\n")
  cat("N =", nrow(df), "\n")
  cat("Events =", sum(df$.event == 1, na.rm = TRUE), "\n")
  print(table(df$LVI_group, useNA = "ifany"))
  print(table(df$LVI_group, df$.event, useNA = "ifany"))
  
  fit <- survfit(Surv(.time, .event) ~ LVI_group, data = df)
  
  p <- ggsurvplot(
    fit,
    data = df,
    risk.table = TRUE,
    pval = TRUE,
    conf.int = FALSE,
    xlab = "Time (years)",
    ylab = paste0(endpoint_name, " probability"),
    legend.title = "LVI label",
    legend.labs = c("LVI-negative", "LVI-positive"),
    title = paste0(endpoint_name, " by pathologist-assessed LVI label"),
    risk.table.height = 0.25,
    ggtheme = theme_bw()
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_full_cohort_LVI_plot.png")),
    plot = p$plot,
    width = 6,
    height = 5,
    dpi = 300
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_full_cohort_LVI_risk_table.png")),
    plot = p$table,
    width = 6,
    height = 2.5,
    dpi = 300
  )
  
  pdf(
    file.path(out_dir, paste0("KM_", endpoint_name, "_by_full_cohort_LVI_combined.pdf")),
    width = 7,
    height = 7
  )
  print(p)
  dev.off()
  
  return(fit)
}

# -------------------------
# 5. Helper function: univariable Cox regression
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
      !is.na(LVI_group)
    )
  
  cox_fit <- coxph(Surv(.time, .event) ~ LVI_group, data = df)
  s <- summary(cox_fit)
  
  result <- tibble(
    endpoint = endpoint_name,
    n = nrow(df),
    events = sum(df$.event == 1, na.rm = TRUE),
    comparison = "LVI-positive vs LVI-negative",
    HR = s$coefficients[1, "exp(coef)"],
    lower_95_CI = s$conf.int[1, "lower .95"],
    upper_95_CI = s$conf.int[1, "upper .95"],
    p_value = s$coefficients[1, "Pr(>|z|)"]
  )
  
  return(result)
}

# -------------------------
# 6. Kaplan-Meier curves
# -------------------------
fit_rfi <- make_km_plot(dat, "RFi_time", "RFi_event", "RFi", out_dir)
fit_drfi <- make_km_plot(dat, "DRFi_time", "DRFi_event", "DRFi", out_dir)
fit_os <- make_km_plot(dat, "OS_time", "OS_event", "OS", out_dir)
fit_bcss <- make_km_plot(dat, "BCSS_time", "BCSS_event", "BCSS", out_dir)

# -------------------------
# 7. Cox regression
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
  file.path(out_dir, "cox_univariable_full_cohort_LVI_results.tsv")
)

cat("\nDone. Outputs saved to:\n")
cat(out_dir, "\n")