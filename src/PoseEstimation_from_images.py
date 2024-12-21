from onnx2torch import convert
import torch
import onnx
import cv2, glob, os, sys
import numpy as np

ESC_KEY = 27
TH = 0.5
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

def main():

    argv = sys.argv
    argc = len(argv)
    
    print('%s <onnx_model> <wildcard for images>' % argv[0])
    
    if argc < 3:   
        quit()
    
    onnx_model = onnx.load(argv[1])
    model = convert(onnx_model)
    model.eval()
   
    base = os.path.basename(argv[1])
    model_name = os.path.splitext(base)[0]
   
    key = -1
   
    paths = glob.glob(argv[2])

    print('Hit ESC_Key to abort')

    for path in paths:

        print('Processing %s' % path)

        base = os.path.basename(path)
        image_name = os.path.splitext(base)[0]
        dst_path = '%s_%s.png' % (image_name, model_name)
    
        img = cv2.imread(path)
        H, W = img.shape[:2]

        dst = img.copy()

        # BGR --> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize 256 x 256
        size = (256,256)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        
        # normalization
        img = img.astype(np.float32) / 255.
        
        # subtracts dataset's mean and normalizes with dataset's standard deviation
        for i in range(3):
            img[:, :, i] = img[:, :, i] - color_mean[i]
            img[:, :, i] = img[:, :, i] / color_std[i]
        
        # channel last [256, 256, 3] --> channel first [3, 256, 256]
        img = img.transpose((2, 0, 1)).astype(np.float32)
        
        # numpy array --> torch tensor
        img = torch.from_numpy(img)
        
        # add batch dimension ([1, 3, 256, 256])
        x = img.unsqueeze(0)
        heatmaps = model(x)[0]
        
        scale_x = W / 64
        scale_y = H / 64
       
        scores = [] 
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            heatmap_numpy = heatmap.detach().numpy()
            idx_1d = np.argmax(heatmap_numpy)
            idx_2d = np.unravel_index(idx_1d, heatmap.shape)
        
            center = (int(idx_2d[1] * scale_x), int(idx_2d[0] * scale_y))
               
            scores.append(heatmap_numpy[idx_2d[0]][idx_2d[1]])
            if scores[-1] > TH:
                dst = cv2.circle(dst, center, 3, (0, 0, 255), -1)
     
        cv2.imwrite(dst_path, dst)
        cv2.imshow('result', dst)
        key = cv2.waitKey(0)

        if key == ESC_KEY:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

