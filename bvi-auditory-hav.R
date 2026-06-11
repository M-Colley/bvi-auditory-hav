if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    setwd(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
}

library(colleyRstats)
colleyRstats::colleyRstats_setup()


library(easystats)
library(ARTool)
library(dplyr)
library(tidyr)
library(ggplot2)
library(FSA)
library(rstatix)


main_df <- readxl::read_xlsx(path = "bvi-auditory-hav.xlsx", sheet = "Results")
main_df <- as.data.frame(main_df)
names(main_df)


# Replace with speaking names
main_df$VIP[main_df$VIP == "N"] <- "Sighted"
main_df$VIP[main_df$VIP == "Y"] <- "BVI"

main_df$UserID <- as.factor(main_df$UserID)
main_df$ConditionID <- as.factor(main_df$ConditionID)
main_df$VIP <- as.factor(main_df$VIP)
main_df$infoContent <- as.factor(main_df$infoContent)


main_df$overallTiATrust <- rowSums(main_df[, c("TiA_Trust1", "TiA_Trust2")]) / 2.0

# Calculate relevant trust scores:
# tiau 2 und 4 inverse
# main_df$TiA_U2 <- 6 - main_df$TiA_U2
# main_df$TiA_U4 <- 6 - main_df$TiA_U4
# main_df$overallTiAUnderstanding <- rowSums(main_df[, c("TiA_U1", "TiA_U2", "TiA_U3", "TiA_U4")]) / 4.0


main_df$ps_score <- rowSums(main_df[, c("perceivedSafety1", "perceivedSafety2", "perceivedSafety3", "perceivedSafety4")]) / 4.0


# SART
main_df$Demand <- main_df$SART1 + main_df$SART2 + main_df$SART3
main_df$Supply <- main_df$SART4 + main_df$SART5 + main_df$SART6 + main_df$SART7
main_df$Understanding <- main_df$SART8 + main_df$SART9 + main_df$SART10

main_df$SA <- main_df$Understanding - (main_df$Demand - main_df$Supply)


# UEQ-S
main_df$pragmatic_quality <- rowSums(main_df[, c("ueqs1", "ueqs2", "ueqs3", "ueqs4")]) / 4.0
main_df$hedonic_quality <- rowSums(main_df[, c("ueqs5", "ueqs6", "ueqs7", "ueqs8")]) / 4.0


labels_xlab <- c("1" = "Low Information", "2" = "Medium", "3" = "High Information")


########### Demographics (Participants section)

demo_df <- readxl::read_xlsx(path = "bvi-auditory-hav.xlsx", sheet = "cleaned")
demo_df <- as.data.frame(demo_df)

demo_df$VIP <- demo_df$visualimpiarment
demo_df$VIP[demo_df$VIP == "N"] <- "Sighted"
demo_df$VIP[demo_df$VIP == "Y"] <- "BVI"
demo_df$VIP <- as.factor(demo_df$VIP)

demo_df$age <- as.numeric(demo_df$age)
demo_df$interest_general <- as.numeric(demo_df$`interest[interest]`)
demo_df$interest_ease <- as.numeric(demo_df$`interest[ease]`)
demo_df$interest_reality <- as.numeric(demo_df$`interest[reality]`)

table(demo_df$gender) # 1 = female, 2 = male
c(M = mean(demo_df$age), SD = sd(demo_df$age), min = min(demo_df$age), max = max(demo_df$age))
table(demo_df$VIP)
table(demo_df$videgree, useNA = "ifany") # A1-A4 = moderate/severe/extreme/blind (answer order)

# general interest in autonomous driving
# not sig. (p=0.331)
c(M = mean(demo_df$interest_general), SD = sd(demo_df$interest_general))
reportMeanAndSD(data = demo_df, iv = "VIP", dv = "interest_general")
wilcox.test(interest_general ~ VIP, data = demo_df)

# expectation that autonomous driving makes life easier
# not sig. (p=0.090) -- group difference is descriptive only
c(M = mean(demo_df$interest_ease), SD = sd(demo_df$interest_ease))
reportMeanAndSD(data = demo_df, iv = "VIP", dv = "interest_ease")
wilcox.test(interest_ease ~ VIP, data = demo_df)

# autonomous driving will become reality within 10 years
# not sig. (p=0.148)
c(M = mean(demo_df$interest_reality), SD = sd(demo_df$interest_reality))
reportMeanAndSD(data = demo_df, iv = "VIP", dv = "interest_reality")
wilcox.test(interest_reality ~ VIP, data = demo_df)


########### TLX
# not sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "tlx_mental", ylab = "Mental Workload", xlabels = labels_xlab)
# ggsave("plots/mental_workload_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "tlx_mental", factors = c("infoContent", "VIP"))

artmodel <- art(formula = tlx_mental ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "cognitive load")

reportMeanAndSD(data = main_df, iv = "VIP", dv = "tlx_mental")

main_df |> ggplot() +
  aes(x = infoContent, y = tlx_mental, fill = VIP, colour = VIP, group = VIP) +
  scale_color_see() +
  theme(legend.position.inside = c(0.7, 0.9)) +
  ylab("Cognitive Load") +
  xlab("Information Content") +
  stat_summary(fun = mean, geom = "point", size = 4.0) +
  # stat_summary(fun = mean, geom = "point", size = 4.0,  aes(group = 1)) +
  # stat_summary(fun = mean, geom = "line", linewidth = 1, linetype = "dashed", aes(group = 1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 2) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .1, position = position_dodge(width = .05)) # 95 % mean_cl_boot is 95% confidence intervals
ggsave("plots/tlx_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


########### Trust
# sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "overallTiATrust", ylab = "Trust", xlabels = labels_xlab)
# ggsave("plots/trust_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "overallTiATrust", factors = c("infoContent", "VIP"))
reportMeanAndSD(data = main_df, iv = "infoContent", dv = "overallTiATrust")

# post-hoc reported in the paper (p_adj = 0.017 / 0.013)
dunnTest(overallTiATrust ~ infoContent, data = main_df, method = "holm")

artmodel <- art(formula = overallTiATrust ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "trust")


########### SART
# sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "SA", ylab = "Situation Awareness", xlabels = labels_xlab)
# ggsave("plots/SART_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "SA", factors = c("infoContent", "VIP"))

anova_test(data = main_df, dv = SA, wid = UserID, within = c(infoContent), between = c(VIP))

dunnTest(SA ~ infoContent, data = main_df, method = "holm")

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "SA")


# Demand
checkAssumptionsForAnova(data = main_df, y = "Demand", factors = c("infoContent", "VIP"))

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "Demand")

artmodel <- art(formula = Demand ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "Demand")

dunnTest(Demand ~ infoContent, data = main_df, method = "holm")


# Supply
checkAssumptionsForAnova(data = main_df, y = "Supply", factors = c("infoContent", "VIP"))

# not sig.
anova_test(data = main_df, dv = Supply, wid = UserID, within = c(infoContent), between = c(VIP))

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "Supply")


# Understanding
checkAssumptionsForAnova(data = main_df, y = "Understanding", factors = c("infoContent", "VIP"))

# sig
dunnTest(Understanding ~ infoContent, data = main_df, method = "holm")

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "Understanding")


artmodel <- art(formula = Understanding ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "Understanding")


########### Perceived Safety
# almost
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "ps_score", ylab = "Perceived Safety", xlabels = labels_xlab)
# ggsave("plots/perceived_safety_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "ps_score", factors = c("infoContent", "VIP"))

artmodel <- art(formula = ps_score ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "perceived safety")

dunnTest(ps_score ~ infoContent, data = main_df, method = "holm")


main_df |> ggplot() +
  aes(x = infoContent, y = ps_score, fill = VIP, colour = VIP, group = VIP) +
  scale_color_see() +
  theme(legend.position.inside = c(0.85, 0.30)) +
  ylab("Perceived Safety") +
  xlab("Information Content") +
  stat_summary(fun = mean, geom = "point", size = 4.0) +
  # stat_summary(fun = mean, geom = "point", size = 4.0,  aes(group = 1)) +
  # stat_summary(fun = mean, geom = "line", linewidth = 1, linetype = "dashed", aes(group = 1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 2) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .1, position = position_dodge(width = .05)) # 95 % mean_cl_boot is 95% confidence intervals
ggsave("plots/ps_score_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


########### UEQ-S
# sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "pragmatic_quality", ylab = "Pragmatic Quality", xlabels = labels_xlab)
# ggsave("plots/pragmatic_quality_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "pragmatic_quality", factors = c("infoContent", "VIP"))

dunnTest(pragmatic_quality ~ infoContent, data = main_df, method = "holm")

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "pragmatic_quality")

artmodel <- art(formula = pragmatic_quality ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "pragmatic quality")


main_df |> ggplot() +
  aes(x = infoContent, y = pragmatic_quality, fill = VIP, colour = VIP, group = VIP) +
  scale_color_see() +
  theme(legend.position.inside = c(0.85, 0.35)) +
  ylab("Pragmatic Quality") +
  xlab("Information Content") +
  stat_summary(fun = mean, geom = "point", size = 4.0) +
  # stat_summary(fun = mean, geom = "point", size = 4.0,  aes(group = 1)) +
  # stat_summary(fun = mean, geom = "line", linewidth = 1, linetype = "dashed", aes(group = 1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 2) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .1, position = position_dodge(width = .05)) # 95 % mean_cl_boot is 95% confidence intervals
ggsave("plots/pragmatic_quality_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


# sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "hedonic_quality", ylab = "Hedonic Quality", xlabels = labels_xlab)
# ggsave("plots/hedonic_quality_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "hedonic_quality", factors = c("infoContent", "VIP"))

dunnTest(hedonic_quality ~ infoContent, data = main_df, method = "holm")

reportMeanAndSD(data = main_df, iv = "infoContent", dv = "hedonic_quality")

artmodel <- art(formula = hedonic_quality ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "hedonic quality")


########### All relevant information
# sig.
ggwithinstatsWithPriorNormalityCheck(data = main_df, x = "ConditionID", y = "informationCommunica", ylab = "All info present", xlabels = labels_xlab)
# ggsave("plots/informationCommunica_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)


checkAssumptionsForAnova(data = main_df, y = "informationCommunica", factors = c("infoContent", "VIP"))

artmodel <- art(formula = informationCommunica ~ infoContent * VIP + Error(UserID / infoContent), data = main_df) |> anova()
artmodel
reportART(artmodel, dv = "presence of all relevant information")

main_df |> ggplot() +
  aes(x = infoContent, y = informationCommunica, fill = VIP, colour = VIP, group = VIP) +
  scale_color_see() +
  theme(legend.position.inside = c(0.85, 0.35)) +
  ylab("Information Presence") +
  xlab("Information Content") +
  stat_summary(fun = mean, geom = "point", size = 4.0) +
  # stat_summary(fun = mean, geom = "point", size = 4.0,  aes(group = 1)) +
  # stat_summary(fun = mean, geom = "line", linewidth = 1, linetype = "dashed", aes(group = 1)) +
  stat_summary(fun = mean, geom = "line", linewidth = 2) +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .1, position = position_dodge(width = .05)) # 95 % mean_cl_boot is 95% confidence intervals
ggsave("plots/informationCommunica_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


#### Final Fragen ####


main_df_final <- readxl::read_xlsx(path = "bvi-auditory-hav.xlsx", sheet = "Final")
main_df_final <- as.data.frame(main_df_final)
names(main_df_final)

main_df_final$VIP <- as.factor(main_df_final$VIP)

labels_xlab_final <- c("N" = "Sighted", "Y" = "BVI")


# reasonable
# sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "reasonable", ylab = "Reasonable", xlabels = labels_xlab_final)
ggsave("plots/reasonable_ggstats.pdf", width = 12, height = 9, device = cairo_pdf)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "reasonable")

# necessary
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "necessary", ylab = "Necessary", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "necessary")


# wouldUse
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "wouldUse", ylab = "wouldUse", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "wouldUse")


# startEnd
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "startEnd", ylab = "Start/End information", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "startEnd")


# drivingRelated
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "drivingRelated", ylab = "drivingRelated", xlabels = labels_xlab_final)


reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "drivingRelated")

# pedestrian
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "pedestrian", ylab = "Pedestrian crossing information", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "pedestrian")

# route
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "route", ylab = "Route information", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "route")


# unforeseenEvents
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "unforeseenEvents", ylab = "Info about Unforeseen Events", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "unforeseenEvents")

# poi
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "poi", ylab = "Points of interest", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "poi")

# surroundings
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "surroundings", ylab = "Surroundings", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "surroundings")

### In-Vehicle Information System ###


# light
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "light", ylab = "Light", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "light")

# dashboard
# sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "dashboard", ylab = "Dashboard", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "dashboard")

# sound
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "sound", ylab = "Sound", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "sound")

# voice
# not sig.
ggbetweenstatsWithPriorNormalityCheck(data = main_df_final, x = "VIP", y = "voice", ylab = "Voice", xlabels = labels_xlab_final)

reportMeanAndSD(data = main_df_final, iv = "VIP", dv = "voice")
