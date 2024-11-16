import torch
import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError

class AdaptiveMultiH_CLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(AdaptiveMultiH_CLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.res_backbone = self._get_basemodel(base_model)
        self.dim_outs = [128, 128, 128]
        dim_in = self.res_backbone.fc.in_features
        self.fc1 = nn.Sequential(nn.Linear(dim_in, self.dim_outs[0]),
                                      nn.ReLU(),
                                      nn.Linear(self.dim_outs[0], self.dim_outs[0]))


        self.mlp_heads = nn.ModuleList([nn.Sequential(nn.Linear(dim_in, 128),
                                                      nn.ReLU(),
                                                      nn.Linear(128, 128)
                                                      )
                                        for _ in range(3)])


        self.temper_fc = nn.Sequential(nn.Linear(128, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 1),
                                       nn.Sigmoid())

        # add mlp projection head
        self.res_backbone.fc = nn.Identity()


    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.res_backbone(x)
        feature_map_lib = []
        temperature_lib = []

        # similarity_matrixset = []

        # different dimension head mlp
        # feature_map1 = self.fc1(x)
        # temperature1 = self.temper_fc(feature_map1)

        for i in range(3):
            feature_map = self.mlp_heads[i](x)
            temperature = self.temper_fc(feature_map)
            feature_map_lib.append(feature_map)
            temperature_lib.append(temperature)

        # feature_map1 = nn.functional.normalize(feature_map1, dim=1)
        # similarity_matrixset.append(torch.matmul(feature_map1, feature_map1.T))

        # feature_map2 = self.fc2(x)
        # temperature2 = self.temper_fc(feature_map2)
        # feature_map2 = nn.functional.normalize(feature_map2, dim=1)
        # similarity_matrixset.append(torch.matmul(feature_map2, feature_map2.T))
        #
        # feature_map3 = self.fc3(x)
        # temperature3 = self.temper_fc(feature_map3)
        # feature_map3 = nn.functional.normalize(feature_map3, dim=1)
        # similarity_matrixset.append(torch.matmul(feature_map3, feature_map3.T))

        # similarity_matrix3 = torch.stack(similarity_matrixset, dim=-1) # (2BS, 2BS, 3)
        #
        # similarity_matrix = torch.mean(similarity_matrix3, dim=-1) # (2BS, 2BS)
        # similarity_matrix = similarity_matrixset[1]
        # temperature = self.temper_fc(similarity_matrix3)

        return feature_map_lib, temperature_lib
