
import torch
from torch import nn
from attributes_recognition_module.src.attention_module.CBAM import CBAM
from attributes_recognition_module.src.feature_extraction.FeatureExtractionModule import FeatureExtractionModule
#from src.feature_extraction.FeatureExtractionModule2 import FeatureExtractionModule2
from attributes_recognition_module.src.ClassificationModule import ClassificationModule
from attributes_recognition_module.src.ASLLoss import  ASLLoss

class MultiTaskNN(nn.Module):
    def __init__(self, dim, convnext="v1", an="CBAM", device='cuda') -> None:
        super(MultiTaskNN, self).__init__()
        if convnext == "v1":
            self.feature_extraction_module = FeatureExtractionModule('base')
        else:
            raise Exception("Version not supported")
       # elif convnext == "v2":
       #     self.feature_extraction_mode2 = FeatureExtractionModule2('base')
    
        if an == "CBAM":
            self.attention_module_upper_color = CBAM(dim)
            self.attention_module_lower_color = CBAM(dim)
            self.attention_module_gender = CBAM(dim)
            self.attention_module_bag = CBAM(dim)
            self.attention_module_hat = CBAM(dim)
        else:
            raise Exception("Version not supported")
        """ elif an == "HAM":
            self.attention_module_upper_color = HAM(dim)
            self.attention_module_lower_color = HAM(dim)
            self.attention_module_gender = HAM(dim)
            self.attention_module_bag = HAM(dim)
            self.attention_module_hat = HAM(dim) """
        
        self.classification_module_upper_color = ClassificationModule(dim, 11)
        self.classification_module_lower_color = ClassificationModule(dim, 11)

        
        # self.classification_module_gender = ClassificationModule(dim, 1)
        # self.classification_module_bag = ClassificationModule(dim, 1)
        # self.classification_module_hat = ClassificationModule(dim, 1)
        self.classification_module_gender = ClassificationModule(dim, 2)
        self.classification_module_bag = ClassificationModule(dim, 2)
        self.classification_module_hat = ClassificationModule(dim, 2)
        
        gamma_positive_upper = torch.tensor([0,0,0,0,0,0,0,0,0,0,0]).to(device)
        gamma_negative_upper = torch.tensor([1,2,4,2,4,4,4,4,2,2,4]).to(device)
        self.upper_color_loss = ASLLoss(num_classes = 11, gamma_positive = gamma_positive_upper, gamma_negative = gamma_negative_upper)


        gamma_positive_lower = torch.tensor([0,0,0,0,0,0,0,0,0,0,0]).to(device)
        gamma_negative_lower = torch.tensor([1,2,4,2,4,4,4,4,4,4,4]).to(device)
        self.lower_color_loss = ASLLoss(num_classes = 11, gamma_positive = gamma_positive_lower, gamma_negative = gamma_negative_lower)

        #self.gender_loss = ASLLoss(num_classes=1, gamma_positive=0, gamma_negative=1)
        #self.bag_loss = ASLLoss(num_classes=1, gamma_positive=0, gamma_negative=1)
        #self.hat_loss = ASLLoss(num_classes=1, gamma_positive=0, gamma_negative=1)
        gamma_positive_gender = torch.tensor([0,0]).to(device)
        gamma_negative_gender = torch.tensor([1,2]).to(device)
        self.gender_loss = ASLLoss(num_classes = 2, gamma_positive=gamma_positive_gender, gamma_negative=gamma_negative_gender)

        gamma_positive_bag = torch.tensor([0,0]).to(device)
        gamma_negative_bag = torch.tensor([1,2]).to(device)
        self.bag_loss = ASLLoss(num_classes = 2, gamma_positive=gamma_positive_bag, gamma_negative=gamma_negative_bag)

        gamma_positive_hat = torch.tensor([0,0]).to(device)
        gamma_negative_hat = torch.tensor([1,2]).to(device)
        self.hat_loss = ASLLoss(num_classes = 2, gamma_positive=gamma_positive_hat, gamma_negative=gamma_negative_hat)
        
        self.weights = torch.nn.Parameter(torch.ones(5).float())
        
        self.device = device
    
    def forward(self, x):
        x = self.feature_extraction_module(x)
        #add dimentsio to x
        x = x.unsqueeze(2).unsqueeze(3)
        
        upper_color_nn = self.attention_module_upper_color(x)
        lower_color_nn = self.attention_module_lower_color(x)
        gender_nn = self.attention_module_gender(x) 
        bag_nn = self.attention_module_bag(x)
        hat_nn = self.attention_module_hat(x)
        
        upper_color_flatten = upper_color_nn.view(x.size(0), -1)
        lower_color_flatten = lower_color_nn.view(x.size(0), -1)
        gender_flatten = gender_nn.view(x.size(0), -1)
        bag_flatten = bag_nn.view(x.size(0), -1)
        hat_flatten = hat_nn.view(x.size(0), -1)
        
        upper_color = self.classification_module_upper_color(upper_color_flatten)
        lower_color = self.classification_module_lower_color(lower_color_flatten)
        gender = self.classification_module_gender(gender_flatten)
        bag = self.classification_module_bag(bag_flatten)
        hat = self.classification_module_hat(hat_flatten)
        
        return upper_color, lower_color, gender, bag, hat
        
    def compute_loss(self, y_pred, y_true):
        upper_color_labels, lower_color_labels, gender_labels, bag_labels, hat_labels = y_true
        preds_upper_color, preds_lower_color, preds_gender, preds_bag, preds_hat = y_pred
        
        mask_upper_color = (upper_color_labels >= 0).float()
        loss_upper_color = self.upper_color_loss(preds_upper_color, upper_color_labels, mask_upper_color, self.device)

        mask_lower_color = (lower_color_labels >= 0).float()
        loss_lower_color = self.lower_color_loss(preds_lower_color, lower_color_labels, mask_lower_color, self.device)
    
        mask_gender = (gender_labels >= 0).float()
        loss_gender = self.gender_loss(preds_gender, gender_labels, mask_gender, self.device)
        
        mask_bag = (bag_labels >= 0).float()
        loss_bag = self.bag_loss(preds_bag, bag_labels, mask_bag, self.device)

        mask_hat = (hat_labels >= 0).float()
        loss_hat = self.hat_loss(preds_hat, hat_labels, mask_hat, self.device)
        
        return [loss_upper_color, loss_lower_color, loss_gender, loss_bag, loss_hat]
    
    def get_last_shared_layer(self):
        return self.feature_extraction_module.convnext.stages[3][2]