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
                <tr><td> model_15.onnx </td><td> 約 9.0 </td></tr>
                <tr><td> mediapipe pose </td><td> 約 7.0 </td></tr>
            </table>
            <br>
            トップダウン方式なので、画面中央付近の一人の骨格しか抽出されません。<br>
            複数人、画面端付近の人の骨格を抽出したい場合は、前段に人検出器を追加して<br>
            切り出した画像をモデルに渡す必要があります。<br>
        </p>
        <h2>環境構築方法</h2>
        <h3>model_15.onnx</h3>
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
    </body>
