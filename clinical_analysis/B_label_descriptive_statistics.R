# =========================
# Clinical Table 1
# using full-cohort pathologist-assessed LVI label
#
# Output:
# 1. Table 1 by LVI group
# =========================

library(readxl)
library(dplyr)
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
              help = "Clinical Excel file containing LVI and clinical covariables."),
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
# 2. Read data
# -------------------------

df <- read_excel(clin_file, sheet = sheet)


df_table1 <- df %>%
  mutate(
    LVI_group = factor(LVI, levels = c(0, 1),
                       labels = c("LVI negative", "LVI positive")),
    
    N_status = factor(N_pos, levels = c(0, 1),
                      labels = c("N0", "N+")),
    
    ER_status = case_when(
      is.na(er) ~ NA_character_,
      er >= 1 ~ "ER positive",
      er < 1 ~ "ER negative"
    ),
    
    PR_status = case_when(
      is.na(pr) ~ NA_character_,
      pr >= 1 ~ "PR positive",
      pr < 1 ~ "PR negative"
    ),
    
    HER2_status = factor(her2_pos, levels = c(0, 1),
                         labels = c("HER2 negative", "HER2 positive")),
    
    T_stage = factor(T_stage),
    ki67 = as.numeric(ki67)
  )

table1 <- df_table1 %>%
  select(
    LVI_group,
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
    by = LVI_group,
    type = list(
      age ~ "continuous",
      tum_size ~ "continuous",
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

print(table1)
table1 %>%
  as_flex_table() %>%
  save_as_docx(path = file.path(out_dir, "Table1_LVI_clinical_covariables.docx"))

cat("\nDone. Table 1 saved to:\n")
cat(file.path(out_dir, "Table1_LVI_clinical_covariables.docx"), "\n")