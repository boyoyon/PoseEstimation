<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Pose Estimation</center></h1>
        <h2>なにものか？</h2>
        <p>
            画像や映像の中の人の骨格を抽出するプログラムです。<br>
            <img src="images/2.png"><br>
            <img src="images/3.png"><br>
            <img src="images/4.png"><br>
            <img src="images/5.png"><br>
            <img src="images/6.png"><br>
            <br>
            HRNETをチャネルプルーニング(下図の各板を薄くするイメージ)したもので<br>
            GPU無でも、そこそこの速度で動作します。<br>
            <br>
            <img src="images/hrnet.png"><br>
            <br>
            <table border="1">
                <tr><th>モデル</th><th>FPS (i7-7700@3.60GHz)</th></tr>
                <tr><td> model_01.onnx </td><td> 約 3.5 </td></tr>
                <tr><td> model_05.onnx </td><td> 約 4.2 </td></tr>
                <tr><td> model_10.onnx </td><td> 約 5.5 </td></tr>
                <tr><td> model_15.onnx </td><td> 約 9.0 </td></tr>
                <tr><td> mediapipe pose </td><td> 約 7.0 </td></tr>
            </table>
            <br>
            トップダウン方式なので、画面中央付近の一人の骨格しか抽出されません。<br>
            複数人、画面端付近の人の骨格を抽出したい場合は、前段に人検出器を追加して<br>
            切り出した画像をモデルに渡す必要があります。<br>
        </p>
        <h2>環境構築方法</h2>
        <h3>model_**.onnx</h3>
        <p>
            pip install onnx2torch opencv-python<br>
        </p>
        <h3>mediapipe pose</h3>
        <p>
            pip install mediapipe<br>
        </p>
        <h2>使い方</h2>
        <h3>カメラからの映像に対して骨格抽出する場合</h3>
        <p>
            python PoseEstimation_from_camera.py<br>
            python mediapipe_PoseEstimation_from_camera.py<br>
        </p>
        <h3>画像に対して骨格抽出する場合</h3>
        <p>
            python PoseEstimation_from_images.py (人が写った画像へのワイルドカード)<br>
            python mediapipe_PoseEstimation_from_images.py (人が写った画像へのワイルドカード)<br>
            例) python PoseEstimation_from_images.py *.png<br>
        </p>
        <h3>model_01, model_05, model_10を使う場合の前準備</h3>
        <p>
            model_01の場合<br>
            python src\merge.py data\onnx\model_01.onnx<br>
        </p>
        <h3>枝刈りしてない HRNET を試す場合</h3>
        <p>
            PINTOさんが提供している onnx 形式のモデルを指定することで、枝刈り前の(重いけど高精度な)HRNETを試すことができます。<br>
            <a href="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/271_HRNet/resources.tar.gz">resources.tar.gz</a>をダウンロードする。<br>
            resources.tar.gzを解凍する。<br>
            resources.tarを解凍する。<br>
            以下のonnx形式のモデルが解凍されます。<br>
            ・hrnet_coco_w32_256x192.onnx<br>
            ・hrnet_coco_w32_Nx256x192.onnx<br>
            ・hrnet_coco_w48_384x288.onnx<br>
            ・hrnet_coco_w48_Nx384x288.onnx<br>
            ・hrnet_mpii_w32_256x256.onnx<br>
            ・hrnet_mpii_w32_Nx256x256.onnx<br>
            <br>
            引数でonnxファイルを指定します。<br>
            python PoseEstimation_from_camera.py <strong>(onnxファイル)</strong><br>
            python PoseEstimation_from_images.py (人が写った画像へのワイルドカード) <strong>(onnxファイル)</strong><br>
            ※ 学習データセット(coco, mpii)によりlandmark(keypoint)が異なります。<br> 
        </p>
    </body>
</html>
