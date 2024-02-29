The complete codes and example data as well as the correct structure of the pyTB is listed as follows:


pyTB/
-readme.txt
-pyTB.yml
-classification_confusing_matrix.py
-preprocess_function/preprocess.py
-submit_version_classification/
    --classification_functions/
        ---classification_model.py
        ---classification_predict.py
        ---classification_train.py
        ---dataLoad.py
    --default_model/
        ---classification_model.h5
        ---classification_model_preprocess.h5
    --models_saved/
    --pred_output/predict_result.csv
    --train_output/
        ---training_process.png
        ---training_process_preprocess.png
    --classification_terinalUI_pred.py
    --classification_terinalUI_train.py

-submit_version_segmentation/
    --segmentation_functions/
        ---dataLoad.py
        ---segmentation_model.py
        ---segmentation_predict.py
        ---segmentation_train.py
    --default_model/segmentation_model.h5
    --models_saved/
    --pred_output/
    --train_output/
        ---accuracy_figures.png
        ---loss_figures.png
        ---predict.png
    --segmentation_terminalUI_pred.py
    segmentation_terminalUI_train.py

-data/
    --mask/*.png
    --mask_img/*.png
    --predData/*.png
    --preprocessed_normal/*.png
    --preprocessed_tb/*.png
    --TB_Chest_Radiography_Database/
        ---Normal/*.png
        ---Tuberculosis/*.png
        ---Normal.metadata.CSV 
        ---Tuberculosis.metadata.CSV 

P.S. To save the time and space, the data file is zipped as data.tar.gz