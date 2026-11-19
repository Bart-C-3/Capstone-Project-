# Theft flags, explained (LSTM + LightGBM ensemble)

## 9584D26951B6E3511F100A9851A929CB  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "9584D26951B6E3511F100A9851A929CB". The decision made by the LSTM model is flagged as possible electricity theft, while the decision made by the LightGBM model is also flagged as possible electricity theft. The language model does not change the decision.

The suspicious periods are identified based on the following criteria:

1. Start date: The start date of the suspicious period is "2015-12-02" to "2015-12-29".
2. End date: The end date of the suspicious period is "2015-12-29" to "2015-12-01".
3. LSTM theft probability if period looked normal: The L

## 0D59BB0AACA78EB9B3DF6A48CB7F94FB  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "0D59BB0AACA78EB9B3DF6A48CB7F94FB". The decision made by the LSTM model is flagged as possible electricity theft. The decision threshold is set at 0.58. The suspicious periods are identified based on the language model, which predicts that there may be a potential theft in the next 29 days. The average daily use in these periods is calculated using the start and end dates of the suspicious periods. The customer's overall average daily use is also calculated, and it is found to be 107.37 kWh. The missing readings in these periods are found to be 3 days. The feature model theft probability is 0.98, indicating that the model has correctly predicted that there may be a potential theft in the next 29 days. The features that

## 67046953A834811A1A0AC209C3E7990C  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "67046953A834811A1A0AC209C3E7990C". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides information about the probability of a suspicious period looking normal. The average daily use in this period (kWh) is 74.85, which is higher than the typical normal customer value (5.48). The missing readings in this period (days) are 28. The feature that pushed towards normal is the use in October 2015 compared with own average (%), which indicates that there was a significant increase in usage during this time frame. The feature that pushed towards theft is the use in October 2016 compared with

## 19B06A8DA87FCC23320D34BCDB304ABE  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "19B06A8DA87FCC23320D34BCDB304ABE". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides additional information about the probability of theft in different periods.

The suspicious periods are identified using the Language Model (LM). The LM predicts the probability of theft in each period based on the customer's overall average daily use, missing readings, and the typical normal customer value. The LM uses a combination of the LSTM model and a LightGBM model to make predictions.

The evidence includes the following features:

- Missing meter readings (days): This feature indicates whether there were any missing readings during the time period being analyzed.

## 13EB464F63702AC64ED58560BA428FEA  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "13EB464F63702AC64ED58560BA428FEA". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides additional information about the probability of theft in different periods.

The suspicious periods are identified using the LSTM model, which predicts the probability of theft in each day based on the previous days' usage data. The average daily use in these periods is calculated using the summary features, which include the highest daily use, average daily use, and missing meter readings. The customer's overall average daily use is also included as a feature.

The most suspicious period is determined by comparing the predicted probability of theft with the customer's overall average daily use. If the

## 5BE3C9C0F9FD91A45AC67BAF60BEB292  (p = 0.99, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range for which the detector was used, the decision made by the LSTM model and a LightGBM model, the language model, and the suspicious periods detected during the time frame.

The first section of the JSON file describes the customer ID, the date range, and the decision made by the LSTM model. The second section describes the language model and the suspicious periods detected during the time frame. The third section describes the LSTM theft probability if the period looked normal, the average daily use in this period, the customer's overall average daily use, the missing readings in this period, and the feature model theft probability.

The fourth section describes the features that pushed towards normal, including the share of days with zero or near-zero use, use in February 2016 compared with own average, and typical normal customer value.

The fifth

## 2898CC3D93544307126F3AA34C02DD9F  (p = 0.99, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range of the suspected theft, the decision made by the LSTM model and a LightGBM model, the language model, and the suspicious periods.

The first section of the JSON file describes the customer ID, which is used to identify the customer who reported the theft. The second section describes the decision made by the LSTM model, which is the probability that the detected theft was due to normal activity. This decision is based on the daily usage curve and summary features.

The third section describes the language model, which uses a combination of the daily usage curve and summary features to predict whether the detected theft was due to normal activity or not. The language model does not change the decision, but it provides additional information to help detect theft.

The fourth section describes the suspicious periods, which are identified using the LSTM model. These periods are when the

## C9A4AFFBB98C526841BF8DEA26EA52E7  (p = 0.99, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range for which the detector was used, the decision made by the LSTM model and a LightGBM model, the language model, and the suspicious periods detected during the time frame.

The first section of the JSON file describes the customer ID, the date range, and the decision made by the LSTM model. The second section describes the language model and the suspicious periods detected during the time frame. The third section describes the LSTM model's decision threshold, the average daily use in this period (kWh), the customer's overall average daily use (kWh), the missing readings in this period (days), and the typical normal customer value. The fourth section describes the features that pushed towards normal, including the use in October 2016 compared with own average (%), the use on Mondays compared with own average (%), and the percentage among normal

## EE64AFF20E6E32EAF33902ADE018C570  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is EE64AFF20E6E32EAF33902ADE018C570, which corresponds to a residential property located in the city of New York. The decision made by the LSTM model is flagged as possible electricity theft, while the decision made by the LightGBM model is confirmed as such. The language model does not change the decision, indicating that it is based on the language used in the data.

The evidence includes three sets of data:

1. Decision: The LSTM model predicts that there is a high probability that the electricity usage in the past few months has been normal, while the LightGBM model predicts that there is a high probability that the electricity usage in the past few months has been unusual.

2. Suspicious periods: These are periods where the LSTM model predicts

## D076FDC891800CA297B7CE4BEF149032  (p = 0.99, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range for which the detector was used, the decision made by the LSTM model and a LightGBM model, the language model, and the suspicious periods detected during the time frame.

The first section of the JSON file describes the customer ID, the date range, and the decision made by the LSTM model. The second section describes the language model and the suspicious periods detected during the time frame. The third section describes the LSTM theft probability if the period looked normal, the average daily use in this period, the customer's overall average daily use, and the missing readings in this period.

The fourth section describes the lstm theft probability if the period looked normal, the average daily use in this period, the customer's overall average daily use, and the missing readings in this period. This section also includes the lstm theft probability if the

## C137E8F3550F81C621BE97152DF95D62  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "C137E8F3550F81C621BE97152DF95D62". The decision made by the LSTM model is flagged as possible electricity theft, while the decision made by the LightGBM model is also flagged as possible electricity theft. The language model does not change the decision.

The suspicious periods are identified based on the following criteria:

1. Start date: The start date of the suspicious period is September 7, 2016.
2. End date: The end date of the suspicious period is October 5, 2016.
3. LSTM theft probability if the period looked normal: The LSTM model predicts that there was a high likelihood of electricity theft during this period.
4. Average daily use in this

## 9B024682953957B760F7494E49E19DE4  (p = 0.99, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "9B024682953957B760F7494E49E19DE4" and the decision made by the AI model is flagged as possible electricity theft. The decision threshold is set at 0.58, and the suspicious periods are identified based on the language model and the average daily use curve. The average daily use in these periods is 53.93 kWh, which is higher than the customer's overall average daily use of 29.62 kWh. The missing readings in these periods are 0 days.

The LSTM model used to detect electricity theft is trained using data from January 2016 to March 2016, while the LightGBM model is used to detect summary features. The LSTM model uses a language model to predict the probability

## 837D54985C93D7902649F9AFDCD0B484  (p = 0.98, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "837D54985C93D7902649F9AFDCD0B484". The decision made by the LSTM model is flagged as possible electricity theft, while the decision made by the LightGBM model is also flagged as possible electricity theft. The probability of theft is calculated using the LSTM model, which is 0.98. The average daily use in the last 90 days is 1247.27 kWh, which is higher than the typical normal customer value of 5.41 kWh. The largest one-day drop in use is 4878.0 kWh, which is higher than the typical normal customer value of 11.66 kWh. The feature that pushed towards normal is the similarity of each day to the day before (correlation),

## A339AE08F6CC2375284A87F319707555  (p = 0.98, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the decision made by the Average of an LSTM model and a LightGBM model, the probability of electricity theft if the period looked normal, the average daily use in the last 90 days, the largest one-day drop in use, and the features that pushed towards normal.

The first section of the JSON file describes the customer ID, the decision made by the Average of an LSTM model and a LightGBM model, and the probability of electricity theft if the period looked normal. The second section describes the average daily use in the last 90 days, the largest one-day drop in use, and the features that pushed towards normal.

The third section of the JSON file describes the features that pushed towards normal, including the use on Mondays compared with own average (%), the use in February 2016 compared with own average (%

## 09D862DE78D8C9A051B09FA4C58CDC25  (p = 0.98, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range of the investigation, the decision made by the AI model, the decision threshold, the suspicious periods, the theft probability, the feature models that pushed towards normalcy, and the features that pushed towards theft.

The first section of the JSON file provides the customer ID, which is used to identify the customer who requested the investigation. The second section describes the decision made by the AI model, which is flagged as possible electricity theft based on the analysis of the daily usage curve and summary features. The third section details the suspicious periods, where the theft probability increases significantly. The fourth section explains the theft probability if the period was looked normal, while the fifth section describes the missing readings in the period.

The next section describes the feature models that pushed towards normalcy, including the highest daily use, average daily use, and largest one-day drop in use

## 694AD0174003D59B2A0BA76DB3BB95AF  (p = 0.98, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "694AD0174003D59B2A0BA76DB3BB95AF". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides additional information about the probability of theft in different periods.

The suspicious periods are identified using the Language Model (LM). The LM predicts the probability of theft in each period based on the customer's overall average daily use, missing readings, and the number of days without readings. The LM uses a combination of LSTM and a LightGBM model to achieve high accuracy.

The evidence includes three sets of data:

1. Decision Threshold: The LM predicts that there is a high probability of theft in the next

## DCA2435A43810553D8E4CBA5FED69987  (p = 0.98, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the decision made by the LSTM model and a LightGBM model, the probability of electricity theft based on the daily usage curve and summary features, the suspicious periods, the average daily use in these periods, the missing readings in these periods, and the probability of electricity theft if the period looked normal.

The first section of the JSON file describes the customer ID, the decision made by the LSTM model, and the probability of electricity theft based on the daily usage curve and summary features. The second section describes the suspicious periods, the average daily use in these periods, the missing readings in these periods, and the probability of electricity theft if the period looked normal.

The third section describes the features that pushed the customer toward normal behavior, such as the similarity of each day to the same day a week before (correlation), the use on Fridays

## AE971C561A62FA8888237ACFCF072036  (p = 0.98, true label 1, written by llm)

The evidence JSON file contains information about an electricity theft detector that has been installed at a customer's home. The data includes the customer ID, the date range of the analysis, the decision made by the AI model, the decision threshold, the suspicious periods, the average daily use in the last 90 days, the customer's overall average daily use, the missing readings in the last 90 days, the feature model theft probability, the features that pushed towards normal, and the features that pushed towards theft.

The first section of the JSON file provides the customer ID, which is used to identify the specific customer whose data is being analyzed. The second section describes the decision made by the AI model, which is flagging the customer as a possible electricity theft suspect based on the analysis of their data. The third section details the decision threshold, which is the minimum percentage of days during which there were no suspicious activity detected. The fourth section describes the suspicious periods, which are the days when there was a significant

## 09120FE02B08469FD40098BB4FE13B68  (p = 0.98, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "09120FE02B08469FD40098BB4FE13B68". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides information about the probability that the detected pattern is due to normal usage. The suspicious periods are identified using a combination of LSTM and LightGBM models. The average daily use in these periods is 33.87 kWh, which is higher than the customer's overall average daily use of 18.55 kWh. Missing readings were found in three periods, with the first one being in September 2016. The feature that pushed towards normal was highest daily use (kWh), with the typical normal customer having a value of 2

## 72F5CFD9A1B8514D9CDC4AE58DBADD21  (p = 0.98, true label 1, written by llm)

The evidence provided in the JSON file is a summary of the results obtained from an electricity theft detector. The customer ID is "72F5CFD9A1B8514D9CDC4AE58DBADD21". The decision made by the LSTM model is flagged as possible electricity theft based on the analysis of daily usage curves and summary features. The language model does not change the decision, but it provides additional information about the probability of theft in different periods.

The suspicious periods are identified using the LSTM model, which predicts the probability of theft in a given period if the normal pattern is followed. The average daily use in this period (kWh) is calculated using the customer's overall average daily use (kWh) and missing readings in this period (days). The feature that pushed towards normal is the median daily use (kWh), which is calculated using the customer's use in October 2016 compared with their own average (%), and the

