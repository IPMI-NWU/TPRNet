from torch import nn
import json 
from dm.get_network import get_network_from_plans
from utils.fore_extraction import getForeCoord,getSquare
from utils.extract_centerline import extract_centerline
from utils.crop_centerline import sample_centerline_and_crop
import torch.nn.functional as F
#import os,sys
#import math
#parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, parent_dir)
import numpy as np
class DyNet(nn.Module):
    def __init__(self, input_c=1, output_c=2, square_size=512,local_layer=7,patch_size=96,focus_layer=7,step=96,centerline_ratio=0.3):
        super(DyNet, self).__init__()
        self.square_size=square_size
        self.patch_size=patch_size
        self.step=step
        self.centerline_ratio=centerline_ratio
        global_arch_kwargs = json.load(open('./dm/config/global.json', 'r'))
        local_arch_kwargs = json.load(open('./dm/config/local_'+str(local_layer)+'.json', 'r'))
        focus_arch_kwargs = json.load(open('./dm/config/focus_'+str(focus_layer)+'.json', 'r'))
        self.global_net=get_network_from_plans(global_arch_kwargs,input_c,output_c)
        self.local_net=get_network_from_plans(local_arch_kwargs,input_c,output_c)
        self.focus_net=get_network_from_plans(focus_arch_kwargs,input_c,output_c)
        # model_state_dict=self.focus_net.state_dict()
        # focus_state_dict=torch.load("/home/gaozhizezhang/jzproject/little_net/run/epochs=100,batch_size=1,local_layers=5,square_size=448/1_experiment_20241216_190340/model_best.pth.tar")
        # for key, param in focus_state_dict['network_weights'].items():
            # if key.replace("focus_net.","") in model_state_dict.keys():
                # model_state_dict[key.replace("focus_net.","")].copy_(param)  # 将加载的参数拷贝到模型中
            # else:
                # print(f"警告：参数 {key} 在模型中未找到!")
        # for param in self.focus_net.parameters():
            # param.requires_grad = False
    def forward(self, x,epoch):
        global_out=self.global_net(x)
        
        x1=F.softmax(global_out,1)
        x1=x1[:,1]
        x1[x1>0.5]=1
        x1[x1<=0.5]=0
        start,end=getForeCoord(x1)
        start,length=getSquare(start,end,self.square_size)
        x_local=x.clone()
        x_local=x_local[:,:,start[0]:start[0]+length,start[1]:start[1]+length]
        local_out=self.local_net(x_local)
        if epoch==0:
            return global_out,local_out,start,length
        global_local_output=global_out.clone()
        global_local_output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]=(global_local_output[:,:,start[0]:start[0]+length,start[1]:start[1]+length]+local_out)/2
        x2=F.softmax(global_local_output,1)
        x2=x2[:,1]
        x2[x2>self.centerline_ratio]=255
        x2[x2<=self.centerline_ratio]=0
        
        centerline=extract_centerline(x2.detach().cpu().numpy())
        x_focus=x.clone()
        x_focus,x_focus_coord=sample_centerline_and_crop(centerline,x_focus,self.patch_size,self.step)
        #print(x_focus.shape)
        focus_output=self.focus_net(x_focus)
        focus_output_coord=[]
        for i in range(len(x_focus_coord)):
            focus_output_coord.append((focus_output[i].unsqueeze(0),x_focus_coord[i]))
        return global_out,local_out,focus_output_coord,start,length
        
from PIL import Image
import numpy as np
import torch
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = image / 255.0
    image = image.unsqueeze(0)
    return image
if __name__=="__main__":
    image_path = "/home/gaozhizezhang/jzproject/data/DeepGlobe/images/100.png"
    image = load_image(image_path)
    model = DyNet(input_c=3, output_c=1)
    model.eval()

    with torch.no_grad():
        output = model(image)
    print("Output shape:", output.shape)
    