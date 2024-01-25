import unittest
from api import test_model
import model_utils
import json
import os
import copy
import pandas as pd
import numpy as np
import shutil

single_data = {"x0": "-0.675304", "x1": "0.137379", "x2": "4.393917364", "x3": "-0.020123474", "x4": "-0.475618592", "x5": "sunday", "x6": "0.157397", \
                "x7": "55.677997", "x8": "1.83605", "x9": "0.91846", "x10": "14.351465", "x11": "nan", "x12": "$3709.93", "x13": "0.819808", "x14": "17.07728", \
                "x15": "-0.243366", "x16": "0.061937", "x17": "14.332908999999999", "x18": "-19.662144", "x19": "0.165622", "x20": "0.146025", "x21": "-2.414621", \
                "x22": "0.353511", "x23": "3.190204", "x24": "-118.124909", "x25": "0.90281", "x26": "0.79805", "x27": "0.5203300000000001", "x28": "14.054438000000001", \
                "x29": "0.871179", "x30": "5.126021", "x31": "asia", "x32": "0.51033987", "x33": "2.43467728", "x34": "-2.04913613", "x35": "1.23089839", "x36": "0.83152122", \
                "x37": "3.50526038", "x38": "-1.89375171", "x39": "-0.95390232", "x40": "-276.43", "x41": "1526.17", "x42": "-1062.4", "x43": "351.54", \
                "x44": "0.09087572", "x45": "0.13512714", "x46": "-0.027221829", "x47": "-0.401745419", "x48": "-0.7682184759999999", "x49": "-1.477928431", \
                "x50": "0.461940432", "x51": "1.684288945", "x52": "-0.628094413", "x53": "0.00528862", "x54": "0.38612031", "x55": "-0.80454146", "x56": "-0.215346985", \
                "x57": "-1.265547487", "x58": "0.6828697490000001", "x59": "0.7241555059999999", "x60": "-0.11302117699999999", "x61": "-0.716963446", "x62": "-0.552213898", \
                "x63": "45.85", "x64": "3.00265249", "x65": "4.05022364", "x66": "0.17271423", "x67": "14.03430494", "x68": "-20.88886923", "x69": "0.57667473", \
                "x70": "0.1727856", "x71": "2.37700832", "x72": "0.48401779", "x73": "3.01276075", "x74": "-97.81706928", "x75": "nan", "x76": "1.80140824", \
                "x77": "0.20838348", "x78": "14.4178935", "x79": "-2.58655254", "x80": "2.52245981", "x81": "November", "x82": "Male", "x83": "0.557207747", \
                "x84": "1.7763087469999999", "x85": "0.47166523200000005", "x86": "0.789085832", "x87": "-1.061310858", "x88": "-0.850872339", "x89": "0.599991103", \
                "x90": "-0.22179097600000003", "x91": "0.406396", "x92": "0.9239033999999999", "x93": "3.19037208", "x94": "-99.4804139", "x95": "0.65872137", \
                "x96": "1.01721083", "x97": "0.84194747", "x98": "-32.13548212", "x99": "-92.81795904"}

batch_data = [single_data for i in range(20)]

class AdPurchaseModelAPITesting(unittest.TestCase):
    def setUp(self):
        # Set up a test client
        self.app = test_model()
        self.client = self.app.test_client()
    
    def test_valid_single_prediction(self):
        response = self.client.post("/predict", json=single_data)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 201)

    def test_invalid_single_prediction(self):
        temp_data = copy.deepcopy(single_data)
        temp_data["x5"] = "blursday"
        response = self.client.post("/predict", json=temp_data)
        self.assertNotEqual(response.status_code, 201)

    def test_valid_batch_prediction(self):
        response = self.client.post("/predict", json=batch_data)
        data = json.loads(response.get_data(as_text=True))

        self.assertEqual(response.status_code, 201)

    def test_invalid_prediction_size(self):
        temp_data = copy.deepcopy(single_data)
        del temp_data["x0"]
        data = pd.DataFrame([temp_data], columns=temp_data.keys())

        response, code = model_utils.validate_input(data)

        self.assertEqual(code, 410)

    def test_invalid_day_of_week(self):
        temp_data = copy.deepcopy(single_data)
        temp_data["x5"] = "tuesblay"

        data = pd.DataFrame([temp_data], columns=temp_data.keys())

        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 420)

    def test_invalid_location(self):
        temp_data = copy.deepcopy(single_data)
        temp_data["x31"] = "peru"

        data = pd.DataFrame([temp_data], columns=temp_data.keys())

        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 421)

    def test_invalid_month(self):
        temp_data = copy.deepcopy(single_data)
        temp_data["x81"] = "Octvember"

        data = pd.DataFrame([temp_data], columns=temp_data.keys())

        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 422)

    def test_invalid_gender(self):
        temp_data = copy.deepcopy(single_data)
        temp_data["x82"] = "Pass"

        data = pd.DataFrame([temp_data], columns=temp_data.keys())

        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 423)

    def test_load_model(self):
        result = model_utils.load_model()

        self.assertTrue(result)

    def test_format_input(self):
        data = pd.DataFrame([single_data], columns=single_data.keys())

        self.assertIsInstance(data.at[0, "x12"], str)
        self.assertIsInstance(data.at[0, "x63"], str)
        formatted_data = model_utils.format_input(data)
        self.assertIsInstance(formatted_data.at[0, "x12"], float)
        self.assertIsInstance(formatted_data.at[0, "x63"], float)

    def test_imputer_input(self):
        data = pd.DataFrame([single_data], columns=single_data.keys())
        data = model_utils.format_input(data)
        data.at[0, "x2"] = np.nan

        # Impute missing fields and scale data, we will use the same fitted objects from training
        self.assertEqual(data.shape[1], 100)
        self.assertTrue(data['x2'].isnull().values.any())
        data_imputed = pd.DataFrame(model_utils.imputer.transform(data.drop(columns=['x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
        self.assertEqual(data_imputed.shape[1], 96)
        self.assertFalse(data_imputed['x2'].isnull().values.any())

    def test_encode_input(self):
        data = pd.DataFrame([single_data], columns=single_data.keys())
        data = model_utils.format_input(data)
        data_imputed = pd.DataFrame(model_utils.imputer.transform(data.drop(columns=['x5', 'x31',  'x81' ,'x82'])), columns=data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
        data_imputed_std = pd.DataFrame(model_utils.std_scaler.transform(data_imputed), columns=data_imputed.columns)
        
        self.assertEqual(data_imputed_std.shape[1], 96)
        self.assertNotIn("x5_monday", data_imputed_std)
        data_imputed_std = model_utils.encode_input(data, data_imputed)
        self.assertEqual(data_imputed_std.shape[1], 124)
        self.assertIn("x5_monday", data_imputed_std)

    def test_validate_list(self):
        self.assertEqual(len(model_utils.variables), 25)

    def test_prepare_model_input(self):
        data = pd.DataFrame([single_data], columns=single_data.keys())

        self.assertEqual(len(data.columns), 100)
        data_pred = model_utils.prepare_model_input(data)

        self.assertEqual(len(data_pred.columns), 25)

    def test_missing_model_file(self):
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        shutil.move("objs/ad_purchase_model.pickle", "tmp/ad_purchase_model.pickle")

        data = pd.DataFrame([single_data], columns=single_data.keys())
        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 510)
        shutil.move("tmp/ad_purchase_model.pickle", "objs/ad_purchase_model.pickle")
        os.rmdir("tmp/")

    def test_missing_variables_file(self):
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        shutil.move("objs/variables.pickle", "tmp/variables.pickle")

        data = pd.DataFrame([single_data], columns=single_data.keys())
        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 511)
        shutil.move("tmp/variables.pickle", "objs/variables.pickle")
        os.rmdir("tmp/")

    def test_missing_imputer_file(self):
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        shutil.move("objs/imputer.pkl", "tmp/imputer.pkl")

        data = pd.DataFrame([single_data], columns=single_data.keys())
        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 512)
        shutil.move("tmp/imputer.pkl", "objs/imputer.pkl")
        os.rmdir("tmp/")

    def test_missing_std_scaler_file(self):
        if not os.path.exists("tmp/"):
            os.makedirs("tmp/")
        shutil.move("objs/std_scaler.pkl", "tmp/std_scaler.pkl")

        data = pd.DataFrame([single_data], columns=single_data.keys())
        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 513)
        shutil.move("tmp/std_scaler.pkl", "objs/std_scaler.pkl")
        os.rmdir("tmp/")

    def test_valid_validate_input(self):
        data = pd.DataFrame([single_data], columns=single_data.keys())
        _, code = model_utils.validate_input(data)

        self.assertEqual(code, 200)

if __name__ == '__main__': # pragma: no cover
    unittest.main()
