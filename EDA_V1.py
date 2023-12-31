{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd55a3f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import pickle\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv(\"cars_sample.csv\", encoding='unicode_escape')\n",
    "\n",
    "# Display information about the dataset\n",
    "df.info()\n",
    "\n",
    "# Drop unnecessary columns\n",
    "df = df.drop(['seller', 'offerType', 'model', 'name'], axis=1)\n",
    "\n",
    "# Calculate and impute null values\n",
    "df['notRepairedDamage'] = df['notRepairedDamage'].fillna(df['notRepairedDamage'].mode()[0])\n",
    "df['vehicleType'] = df['vehicleType'].fillna(df['vehicleType'].mode()[0])\n",
    "df['fuelType'] = df['fuelType'].fillna(df['fuelType'].mode()[0])\n",
    "df['gearbox'] = df['gearbox'].fillna(df['gearbox'].mode()[0])\n",
    "\n",
    "# Separate date and time in dateCrawled, dateCreated & lastSeen\n",
    "df['Date_dateCrawled'] = pd.to_datetime(df['dateCrawled']).dt.date\n",
    "df['Time_dateCrawled'] = pd.to_datetime(df['dateCrawled']).dt.time\n",
    "df = df.drop(['dateCrawled'], axis=1)\n",
    "df['Date_dateCreated'] = pd.to_datetime(df['dateCreated']).dt.date\n",
    "df['Time_dateCreated'] = pd.to_datetime(df['dateCreated']).dt.time\n",
    "df = df.drop(['dateCreated'], axis=1)\n",
    "df['Date_lastSeen'] = pd.to_datetime(df['lastSeen']).dt.date\n",
    "df['Time_lastSeen'] = pd.to_datetime(df['lastSeen']).dt.time\n",
    "df = df.drop(['lastSeen'], axis=1)\n",
    "df = df.drop(['Date_dateCrawled', 'Time_dateCrawled', 'Date_dateCreated', 'Date_lastSeen', 'Time_lastSeen'], axis=1)\n",
    "\n",
    "# Check and handle outliers and incorrect entries in columns\n",
    "df = df[~((df['yearOfRegistration'] >= 2020) | (df['yearOfRegistration'] <= 1900))]\n",
    "df['powerPS'] = np.where((df['powerPS'] > 2400) | (df['powerPS'] < 10), 110, df['powerPS'])\n",
    "df = df[~((df['price'] >= 100000) | (df['price'] == 0))]\n",
    "\n",
    "# Univariate analysis\n",
    "# (Include functions for univariate_cat and univariate_con, then perform analysis for selected columns)\n",
    "\n",
    "# Feature Engineering\n",
    "df = pd.get_dummies(df, columns=['abtest', 'vehicleType', 'fuelType', 'gearbox', 'notRepairedDamage'], drop_first=True)\n",
    "\n",
    "label_encoder = LabelEncoder()\n",
    "df['brand_encoded'] = label_encoder.fit_transform(df['brand'])\n",
    "with open('label_encoder_brand.pkl', 'wb') as label_encoder_file:\n",
    "    pickle.dump(label_encoder, label_encoder_file)\n",
    "\n",
    "df = df.drop('brand', axis=1)\n",
    "df['YearsFromRegistration'] = 2023 - df['yearOfRegistration']\n",
    "df = df.drop('yearOfRegistration', axis=1)\n",
    "\n",
    "bins = [0, 3, 6, 9, 12]\n",
    "labels = [\"Q1\", \"Q2\", \"Q3\", \"Q4\"]\n",
    "df['monthOfRegistration'] = pd.cut(df['monthOfRegistration'], bins=bins, labels=labels)\n",
    "df = pd.get_dummies(df, columns=['monthOfRegistration'], drop_first=True)\n",
    "\n",
    "df = df.drop('postalCode', axis=1)\n",
    "\n",
    "desired_columns = ['YearsFromRegistration', 'powerPS', 'kilometer', 'abtest_test',\n",
    "                   'vehicleType_cabrio', 'vehicleType_coupe', 'vehicleType_limousine',\n",
    "                   'vehicleType_others', 'vehicleType_small car', 'vehicleType_station wagon',\n",
    "                   'vehicleType_suv', 'fuelType_diesel', 'fuelType_electro', 'fuelType_hybrid',\n",
    "                   'fuelType_lpg', 'fuelType_other', 'fuelType_petrol', 'gearbox_manual',\n",
    "                   'notRepairedDamage_yes', 'brand_encoded', 'monthOfRegistration_Q2',\n",
    "                   'monthOfRegistration_Q3', 'monthOfRegistration_Q4', 'price']\n",
    "df = df[desired_columns]\n",
    "\n",
    "# Export dataframe to CSV\n",
    "df.to_csv(\"Data_to_ModelBuilding.csv\", index=False)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
