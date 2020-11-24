%read table from csv file
table = readtable('adult.csv');

%convert string values into categories for easier data manipulation
table.workclass = (categorical(table.workclass));
table.education = (categorical(table.education));
table.marital_status = (categorical(table.marital_status));
table.occupation = (categorical(table.occupation));
table.relationship = (categorical(table.relationship));
table.race = (categorical(table.race));
table.sex = (categorical(table.sex));
table.native_country = (categorical(table.native_country));

%encode label(census_income) into 1 and 0 for easier calculation of p and n 
table.census_income = double(categorical(table.census_income));
table.census_income = table.census_income - 1

%remove variables which have less estimated predictor importance
table = removevars(table, [1 2 3 4 9 10 12 13 14]);
