# =========================
# Clinical Table 1
# using threshold-based model-predicted LVI group
#
# Output:
# 1. Table 1 by model-predicted LVI group
# =========================

library(readr)
library(readxl)
library(dplyr)
library(gtsummary)
library(flextable)
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
              help = "Clinical Excel file with clinical covariables."),
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
# 4. Basic column checks
# -------------------------

required_pred_cols <- c(
  "patient_id",
  "slide_id",
  "y_true",
  "model_score"
)

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
  "her2_pos"
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
# 5. Create model-predicted LVI group
# -------------------------

pred_group <- pred %>%
  mutate(
    patient_id = as.character(patient_id),
    model_score = as.numeric(model_score),
    y_true = as.numeric(y_true),
    
    model_pred_LVI = ifelse(model_score >= selected_threshold, 1, 0),
    
    model_LVI_group = factor(
      ifelse(
        model_pred_LVI == 1,
        "Model-predicted LVI-positive",
        "Model-predicted LVI-negative"
      ),
      levels = c(
        "Model-predicted LVI-negative",
        "Model-predicted LVI-positive"
      )
    )
  ) %>%
  select(
    patient_id,
    slide_id,
    y_true,
    model_score,
    model_pred_LVI,
    model_LVI_group
  )

# Check duplicate patient IDs
if (any(duplicated(pred_group$patient_id))) {
  stop("Duplicated patient_id found in prediction table. Please check patient-slide mapping.")
}

cat("\nModel-predicted LVI group counts:\n")
print(table(pred_group$model_LVI_group, useNA = "ifany"))

# -------------------------
# 6. Merge prediction and clinical data
# -------------------------

dat <- clin %>%
  mutate(patient_id = as.character(patient_id)) %>%
  inner_join(pred_group, by = "patient_id")

cat("\nMerged patient number:", nrow(dat), "\n")

# -------------------------
# 7. Prepare selected clinical variables for Table 1
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
# 8. Table 1 by model-predicted LVI group
# -------------------------

table1_model_pred <- dat_table1 %>%
  select(
    model_LVI_group,
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
    by = model_LVI_group,
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

print(table1_model_pred)

# -------------------------
# 9. Save Table 1 as Word file
# -------------------------

table1_model_pred %>%
  as_flex_table() %>%
  save_as_docx(
    path = file.path(out_dir, "Table1_by_model_predicted_LVI_group.docx")
  )

cat("\nDone. Table 1 saved to:\n")
cat(file.path(out_dir, "Table1_by_model_predicted_LVI_group.docx"), "\n")