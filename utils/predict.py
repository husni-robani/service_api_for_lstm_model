# the way to know path of the project
from pathlib import Path
# print(Path.cwd())

import torch
import sys
# append another directory
sys.path.append('/Users/bani/dev/project_TA/Flask_API/model')
sys.path.append('/Users/bani/dev/project_TA/Flask_API/utils')
# custom dependencies
import model_config as config
import model_architecture
from preprocessing import TextPreprocessing

class Predict():
    def __init__(self):
        self.model = model_architecture.LSTMClassifier(config.input_size, config.hidden_size, config.num_layers, config.num_classes, config.dropout)
        self.model.load_state_dict(torch.load('./model/pretrained_model/lstm_model.pt'))
    def make_predict(self, text):
        # preprocess the text to numerical representation
        preprocessing = TextPreprocessing()
        text = preprocessing.preprocess_text(text)

        # predict with model
        self.model.eval()
        with torch.inference_mode():
            output = self.model(text)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # decode study program code
        for study_program, code in config.studyprogram_encoder.items():
            if predicted_class == code:
                result = study_program
        
        return result, predicted_class, output
