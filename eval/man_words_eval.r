setwd("~/Documents/UCD/Projects/gender_representation/results")

library(dplyr)
library(psych)
library(tidyr)

# man_words <- read.csv("man_words_all_wiki.txt") %>% 
#   na.omit() %>%
#   mutate(gender = as.factor(gender)) %>%
#   mutate(affix = as.factor(affix)) %>%
#   mutate(tag = as.factor(tag))

# summary(man_words)
# 
# man_words %>% group_by(gender, affix) %>%
#   tally()



#data <- read.csv("lexical_gender_wiki1000.csv") %>%
data <- read.csv("lexical_gender_femmasc_wiki1000_majority.csv") %>%
  mutate(tag = as.factor(tag)) %>%
  mutate(wn_label = as.factor(wn_label)) %>%
  mutate(mw_label = as.factor(mw_label)) %>%
  mutate(dc_label = as.factor(dc_label)) %>%
  mutate(comb_label = as.factor(comb_label)) 

summary(data)

data %>% group_by(tag) %>%
  tally()

data %>% group_by(comb_label) %>%
  tally()

data %>% 
  group_by(tag, wn_label) %>%
  tally()

data %>% 
  group_by(tag, mw_label) %>%
  tally()

data %>% 
  group_by(tag, comb_label) %>%
  tally()

data %>% 
  group_by(tag, true_label) %>%
  tally()

only_fem_masc = data %>%
  filter(mw_label %in% c("fem", "masc") | wn_label %in% c("fem", "masc") | 
           dc_label %in% c("fem", "masc"))



write.csv(only_fem_masc, "lexical_gender_femmasc_wiki1000.csv", row.names = FALSE)
