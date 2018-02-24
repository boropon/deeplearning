#!/usr/bin/python3

import sys
from ftplib import FTP

def ftp_put(ip, filename):
    #FTP login
    _ftp = FTP(ip)
    _ftp.login()

    #ファイルオープン
    _file = open(filename, 'rb')

    #FTP put
    _ftp.storbinary("STOR " + filename, _file)

    #ファイルクローズ
    _file.close()

    #FTP quit
    _ftp.quit()

def send_to_mptiff(ip, filename)
    #スーパーオプション filetype=mediaprinttiff指定付きでFTP putを行う
    ftp_put(ip, filename + ' ,filetype=mediaprinttiff')

if __name__ == '__main__':
    args = sys.argv
    ip = argv[1]
    filename = argv[2]
    send_to_mptiff(ip, filename)
