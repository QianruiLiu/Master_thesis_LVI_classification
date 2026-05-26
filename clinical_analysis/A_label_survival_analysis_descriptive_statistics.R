# =========================
# ROI-label clinical association analysis
#
# Aim:
# 1. Association between Pathologist A's label and survival endpoints
#    - KM curves
#    - Univariable Cox regression
#
# 2. Association between Pathologist A's label and clinical characteristics
#    - Table 1 by ROI group
#
# Input files:
# - Clinical table: patient-level clinical and survival data
# - Mapping table: patient_id and slide_id
# - ROI table: slide-level Pathologist A's label from another pathologist
# =========================

library(readxl)
library(readr)
library(dplyr)
library(stringr)
library(tibble)
library(survival)
library(survminer)
library(ggplot2)
library(gtsummary)
library(flextable)
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
              help = "Clinical Excel file with patient-level clinical and survival data."),
  make_option(c("--mapping-file"), dest = "mapping_file", type = "character",
              help = "TSV file containing patient_id and slide_id."),
  make_option(c("--roi-file"), dest = "roi_file", type = "character",
              help = "TSV file containing slide-level Pathologist A ROI label; expected columns include id and ROI."),
  make_option(c("--out-dir"), dest = "out_dir", type = "character",
              help = "Output directory."),
  make_option(c("--sheet"), dest = "sheet", type = "integer", default = 1,
              help = "Excel sheet index or name for the clinical file [default: %default].")
)

opt <- parse_args(OptionParser(option_list = option_list))
check_required_args(opt, c("clin_file", "mapping_file", "roi_file", "out_dir"))

clin_file <- opt$clin_file
mapping_file <- opt$mapping_file
roi_file <- opt$roi_file
out_dir <- opt$out_dir
sheet <- opt$sheet

check_input_file(clin_file, "Clinical file")
check_input_file(mapping_file, "Mapping file")
check_input_file(roi_file, "ROI file")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------------------------
# 2. Read data
# -------------------------

clin <- read_excel(clin_file, sheet = sheet)
mapping <- read_tsv(mapping_file, show_col_types = FALSE)
roi_raw <- read_tsv(roi_file, show_col_types = FALSE)

# -------------------------
# 3. Basic column checks
# -------------------------

required_clin_cols <- c(
  "patient_id",
  "age",
  "tum_size",
  "T_stage",
  "N_pos",
  "N_no_of_pos",
  "er",
  "pr",
  "ki67",
  "her2_pos",
  "fu_survival",
  "surv_outcome",
  "RFi_time",
  "RFi_event",
  "DRFi_time",
  "DRFi_event"
)

required_mapping_cols <- c(
  "patient_id",
  "slide_id"
)

required_roi_cols <- c(
  "id",
  "ROI"
)

missing_clin <- setdiff(required_clin_cols, colnames(clin))
missing_mapping <- setdiff(required_mapping_cols, colnames(mapping))
missing_roi <- setdiff(required_roi_cols, colnames(roi_raw))

if (length(missing_clin) > 0) {
  stop(paste("Missing columns in clinical file:", paste(missing_clin, collapse = ", ")))
}

if (length(missing_mapping) > 0) {
  stop(paste("Missing columns in mapping file:", paste(missing_mapping, collapse = ", ")))
}

if (length(missing_roi) > 0) {
  stop(paste("Missing columns in ROI file:", paste(missing_roi, collapse = ", ")))
}

# -------------------------
# 4. Prepare Pathologist A's label at patient level
# -------------------------

mapping_clean <- mapping %>%
  mutate(
    patient_id = as.character(patient_id),
    slide_id = as.character(slide_id),
    slide_id_clean = str_remove(slide_id, "\\.ndpi$")
  )

roi_clean <- roi_raw %>%
  mutate(
    id = as.character(id),
    slide_id_clean = str_remove(id, "\\.ndpi$"),
    ROI = as.numeric(ROI)
  ) %>%
  select(
    slide_id_clean,
    ROI,
    everything()
  )

roi_patient <- mapping_clean %>%
  inner_join(roi_clean, by = "slide_id_clean") %>%
  mutate(
    ROI_group = factor(
      ifelse(ROI == 1, "ROI-positive", "ROI-negative"),
      levels = c("ROI-negative", "ROI-positive")
    )
  ) %>%
  select(
    patient_id,
    slide_id,
    slide_id_clean,
    ROI,
    ROI_group
  )

# Check duplicate patient IDs after mapping
if (any(duplicated(roi_patient$patient_id))) {
  duplicated_ids <- roi_patient$patient_id[duplicated(roi_patient$patient_id)]
  stop(
    paste(
      "Duplicated patient_id found after ROI mapping. Please check patient-slide mapping:",
      paste(unique(duplicated_ids), collapse = ", ")
    )
  )
}

cat("\nPathologist A's label counts after patient-slide mapping:\n")
print(table(roi_patient$ROI_group, useNA = "ifany"))

# -------------------------
# 5. Merge clinical data with Pathologist A's label
# -------------------------

dat <- clin %>%
  mutate(patient_id = as.character(patient_id)) %>%
  inner_join(roi_patient, by = "patient_id") %>%
  filter(!is.na(ROI))

cat("\nMerged patient number with Pathologist A's label:", nrow(dat), "\n")

cat("\nROI group counts in merged data:\n")
print(table(dat$ROI_group, useNA = "ifany"))

write_tsv(
  dat,
  file.path(out_dir, "merged_clinical_ROI_label_data.tsv")
)

# ============================================================
# Part 1. Pathologist A's label vs survival analysis
# ============================================================

# -------------------------
# 6. Prepare survival endpoints
# -------------------------

dat_surv <- dat %>%
  mutate(
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
  )

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
      !is.na(ROI_group)
    )
  
  cat("\n=========================\n")
  cat(endpoint_name, "\n")
  cat("=========================\n")
  cat("N =", nrow(df), "\n")
  cat("Events =", sum(df$.event == 1, na.rm = TRUE), "\n")
  print(table(df$ROI_group, useNA = "ifany"))
  print(table(df$ROI_group, df$.event, useNA = "ifany"))
  
  fit <- survfit(Surv(.time, .event) ~ ROI_group, data = df)
  
  p <- ggsurvplot(
    fit,
    data = df,
    risk.table = TRUE,
    pval = TRUE,
    conf.int = FALSE,
    xlab = "Time (years)",
    ylab = paste0(endpoint_name, " probability"),
    legend.title = "Pathologist A's label",
    legend.labs = c("ROI-negative", "ROI-positive"),
    title = paste0(endpoint_name, " by Pathologist A's label"),
    risk.table.height = 0.25,
    ggtheme = theme_bw()
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_ROI_label_plot.png")),
    plot = p$plot,
    width = 6,
    height = 5,
    dpi = 300
  )
  
  ggsave(
    filename = file.path(out_dir, paste0("KM_", endpoint_name, "_by_ROI_label_risk_table.png")),
    plot = p$table,
    width = 6,
    height = 2.5,
    dpi = 300
  )
  
  pdf(
    file.path(out_dir, paste0("KM_", endpoint_name, "_by_ROI_label_combined.pdf")),
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
      !is.na(ROI_group)
    )
  
  if (nlevels(droplevels(df$ROI_group)) != 2) {
    return(
      tibble(
        endpoint = endpoint_name,
        n = nrow(df),
        events = sum(df$.event == 1, na.rm = TRUE),
        comparison = "ROI-positive vs ROI-negative",
        HR = NA_real_,
        lower_95_CI = NA_real_,
        upper_95_CI = NA_real_,
        p_value = NA_real_
      )
    )
  }
  
  cox_fit <- coxph(Surv(.time, .event) ~ ROI_group, data = df)
  s <- summary(cox_fit)
  
  tibble(
    endpoint = endpoint_name,
    n = nrow(df),
    events = sum(df$.event == 1, na.rm = TRUE),
    comparison = "ROI-positive vs ROI-negative",
    HR = s$coefficients[1, "exp(coef)"],
    lower_95_CI = s$conf.int[1, "lower .95"],
    upper_95_CI = s$conf.int[1, "upper .95"],
    p_value = s$coefficients[1, "Pr(>|z|)"]
  )
}

# -------------------------
# 9. Kaplan-Meier curves
# -------------------------

fit_rfi <- make_km_plot(dat_surv, "RFi_time", "RFi_event", "RFi", out_dir)
fit_drfi <- make_km_plot(dat_surv, "DRFi_time", "DRFi_event", "DRFi", out_dir)
fit_os <- make_km_plot(dat_surv, "OS_time", "OS_event", "OS", out_dir)
fit_bcss <- make_km_plot(dat_surv, "BCSS_time", "BCSS_event", "BCSS", out_dir)

# -------------------------
# 10. Cox regression
# -------------------------

cox_results <- bind_rows(
  run_cox(dat_surv, "RFi_time", "RFi_event", "RFi"),
  run_cox(dat_surv, "DRFi_time", "DRFi_event", "DRFi"),
  run_cox(dat_surv, "OS_time", "OS_event", "OS"),
  run_cox(dat_surv, "BCSS_time", "BCSS_event", "BCSS")
)

print(cox_results)

write_tsv(
  cox_results,
  file.path(out_dir, "cox_univariable_ROI_label_results.tsv")
)

# ============================================================
# Part 2. Pathologist A's label vs clinical characteristics
# ============================================================

# -------------------------
# 11. Prepare clinical variables for Table 1
# -------------------------

dat_table1 <- dat %>%
  mutate(
    age = as.numeric(age),
    tum_size = as.numeric(tum_size),
    N_no_of_pos = as.numeric(N_no_of_pos),
    ki67 = as.numeric(ki67),
    er = as.numeric(er),
    pr = as.numeric(pr),
    
    T_stage = factor(
      T_stage,
      levels = c(1, 2, 3)
    ),
    
    N_status = factor(
      N_pos,
      levels = c(0, 1),
      labels = c("N0", "N+")
    ),
    
    ER_status = case_when(
      is.na(er) ~ NA_character_,
      er >= 1 ~ "ER positive",
      er < 1 ~ "ER negative"
    ),
    
    ER_status = factor(
      ER_status,
      levels = c("ER negative", "ER positive")
    ),
    
    PR_status = case_when(
      is.na(pr) ~ NA_character_,
      pr >= 1 ~ "PR positive",
      pr < 1 ~ "PR negative"
    ),
    
    PR_status = factor(
      PR_status,
      levels = c("PR negative", "PR positive")
    ),
    
    HER2_status = factor(
      her2_pos,
      levels = c(0, 1),
      labels = c("HER2 negative", "HER2 positive")
    )
  )

# -------------------------
# 12. Table 1 by Pathologist A's label
# -------------------------

table1_roi <- dat_table1 %>%
  select(
    ROI_group,
    age,
    tum_size,
    T_stage,
    N_status,
    N_no_of_pos,
    ER_status,
    PR_status,
    ki67,
    HER2_status
  ) %>%
  tbl_summary(
    by = ROI_group,
    type = list(
      age ~ "continuous",
      tum_size ~ "continuous",
      N_no_of_pos ~ "continuous",
      ki67 ~ "continuous"
    ),
    statistic = list(
      all_continuous() ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    missing = "ifany"
  ) %>%
  add_p(
    test = list(
      all_continuous() ~ "wilcox.test",
      all_categorical() ~ "fisher.test"
    )
  ) %>%
  add_overall() %>%
  bold_labels()

print(table1_roi)

table1_roi %>%
  as_flex_table() %>%
  save_as_docx(
    path = file.path(out_dir, "Table1_by_ROI_label_clinical_covariables.docx")
  )

cat("\nDone. Outputs saved to:\n")
cat(out_dir, "\n")