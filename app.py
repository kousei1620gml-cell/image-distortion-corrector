from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.route('/')
def index():
    # templates/index.html を表示する
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # 1. ブラウザから送られてきた画像データを読み込む
        file = request.files['image']
        npimg = np.fromfile(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 2. 四隅の座標と、選択された比率を受け取る
        points = json.loads(request.form['points'])
        ratio_w = float(request.form['ratio_w'])
        ratio_h = float(request.form['ratio_h'])

        # 3. 射影変換（ゆがみ補正）の処理
        pts = np.array(points, dtype="float32")
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        maxHeight = int(maxWidth * (ratio_h / ratio_w))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # 4. 補正した画像をBase64（テキスト）に変換してブラウザに返す
        _, buffer = cv2.imencode('.jpg', warped)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'success': True, 'image': img_base64})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # サーバー起動！
    app.run(debug=True)