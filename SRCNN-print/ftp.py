from ftplib import FTP

def ftp_put(ip, filename, superoption):
    """
    FTPのopen->put->quitを行う
    """
    # FTP login
    _ftp = FTP(ip)
    _ftp.login()

    # ファイルオープン
    _file = open(filename, 'rb')

    # FTP put
    _ftp.storbinary('STOR ' + filename + ' ,' + superoption, _file)

    # ファイルクローズ
    _file.close()

    # FTP quit
    _ftp.quit()

def send_to_mptiff(ip, filename, supopt):
    """
    スーパーオプション filetype=mediaprinttiff指定付きでFTP putを行う
    """
    ftp_put(ip, filename, "filetype=mediaprinttiff," + supopt)
