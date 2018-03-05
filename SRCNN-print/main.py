import sys
import cv2
import ftp
import subprocess

def get_paper_size(paper_name):
    """
    引数で渡された用紙の幅、高さ(単位は0.1mm)を返す
    """
    paper_dict = {"A4":(2970, 2100), "A3":(4200, 2970)}
    return paper_dict[paper_name]

def calc_magnification(height, width, paper_name):
    """
    用紙に画像をfitさせるための変倍率を返す
    """
    # 画像の長辺短辺の長さ
    if height > width:
        image_longedge = height
        image_shortedge = width
    else:
        image_longedge = width
        image_shortedge = height

    # 用紙の長辺短辺の長さ
    paper_height, paper_width = get_paper_size(paper_name)
    reso = 300
    if paper_height > paper_width:
         paper_longedge = paper_height * reso / 254
         paper_shortedge = paper_width * reso / 254
    else:
         paper_longedge = paper_width * reso / 254
         paper_shortedge = paper_height * reso / 254

    # 変倍率計算
    long_ratio = paper_longedge / image_longedge
    short_ratio = paper_shortedge / image_shortedge
    if long_ratio < short_ratio:
        ratio = long_ratio
    else:
        ratio = short_ratio

    # 縮小変倍はしない(等倍とする)
    if ratio < 1:
        ratio = 1
    
    # 3倍以上に変倍しない
    if ratio > 3:
        ratio = 3

    return ratio

def load_image(filename):
    """
    画像を読み込んで、画像オブジェクト、高さ、幅、チャンネル数を返す
    """
    # 画像の読み込み
    img = cv2.imread(filename)
    if img is None:
        print("Failed to load image file.")
        sys.exit(1)

    # 画像の幅、高さ、チャンネル数の取得
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        print("Grayscale is not supported.")
        sys.exit(1)
        #height, width = img.shape[:2]
        #channels = 1
    
    return (img, height, width, channels)

if __name__ == '__main__':
    """
    引数1 - プリンタのIPアドレス
    引数2 - 印刷する画像ファイル
    """
    args = sys.argv
    ip = args[1]
    filename = args[2]
    
    # 印刷する画像を読み込み
    img, height, width, channels = load_image(filename)

    # 印刷する画像を用紙にfitするサイズに変倍
    resized_filename = 'resized_image.jpg'
    mag = calc_magnification(height, width, "A4")
    resized_img = cv2.resize(img, (round(width*mag), round(height*mag)), interpolation=cv2.INTER_CUBIC)
    #resized_img = cv2.resize(img, (round(width*mag), round(height*mag)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(resized_filename, resized_img)
    
    # 変倍した画像に対して超解像処理を実施
    cmd = "python3 SRCNN/main.py --is_train=False --test_img " + resized_filename + " --scale 1"
    subprocess.call(cmd.strip().split(" "))

    # 超解像処理をかけたファイルをプリンタへ送信
    ftp.send_to_mptiff(ip, "result/result.jpg", "tray=tray1,printscaling=fit")

    # 超解像処理をかける前のファイルをプリンタへ送信(比較用)
    #ftp.send_to_mptiff(ip, "resized_image.jpg", "tray=tray1,printscaling=fit")
