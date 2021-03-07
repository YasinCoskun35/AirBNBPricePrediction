import pandas as pd
def clean_data():
    #File Readed
    df = pd.read_csv ("listings_filled.csv")
    #Columns will we use 
    willToUseColumns=['price',
    'minimum_nights',
    'maximum_nights',
    'host_is_superhost',
    'neighbourhood_cleansed',
    'room_type',
    'accommodates',
    'bathrooms_text',
    'bedrooms',
    'beds',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'instant_bookable',
    'calculated_host_listings_count']
    deleteColumns=[]
    #Deleting proccess
    for column in df.columns:
        if willToUseColumns.count(column)<1:
            deleteColumns.append(column)

    df=df.drop(columns=deleteColumns)
    print(len(df.columns))

    df.to_csv("listings_filled.csv")