from onnx2torch import convert
import torch
import onnx
import cv2, os, sys
import numpy as np

ESC_KEY = 27
TH = 0.5
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

def main():

    argv = sys.argv
    argc = len(argv)
    
    if argc < 2:
        print('%s <onnx_model>' % argv[0])
        quit()
    
    onnx_model = onnx.load(argv[1])
    model = convert(onnx_model)
    model.eval()
    
    cap = cv2.VideoCapture(0)
    
    key = -1
    
    freq = cv2.getTickFrequency()
    prev = cv2.getTickCount()
    frameNo = 0
    
    while key != ESC_KEY: 
    
        ret, frame = cap.read()
    
        if not ret:
            continue 
    
        dst = frame.copy()
    
        # BGR --> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # resize 256 x 256
        size = (256,256)
        img = cv2.resize(img_rgb, size, interpolation=cv2.INTER_CUBIC)
        
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
        
        meter = cv2.TickMeter()
        meter.start()
        heatmaps = model(x)[0]
        meter.stop()
       
        #print(meter.getTimeSec())
        
        scale_x = frame.shape[1] / 64
        scale_y = frame.shape[0] / 64
       
        scores = [] 
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            heatmap_numpy = heatmap.detach().numpy()
            idx_1d = np.argmax(heatmap_numpy)
            idx_2d = np.unravel_index(idx_1d, heatmap.shape)
        
            center = (int(idx_2d[1] * scale_x), int(idx_2d[0] * scale_y))
               
            scores.append(heatmap_numpy[idx_2d[0]][idx_2d[1]])
            if scores[-1] > TH:
                dst = cv2.circle(dst, center, 5, (0, 0, 255), -1)
        
        #print(scores)
        cv2.imshow('result', dst)
        
        key = cv2.waitKey(10)
    
        frameNo += 1
        curr = cv2.getTickCount()
        elapse = (curr - prev) / freq
        if elapse > 1:
            print('%.1f fps' % (frameNo / elapse))
            frameNo = 0
            prev= curr
    
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

