#!/usr/bin/python3

from ftplib import FTP

def main():

    ip = '10.247.83.78'
    _ftp = FTP(ip)
    _ftp.login()
    
    filename = 'mmr.tif'
    _file = open(filename, 'rb')
    _ftp.storbinary("STOR " + filename + ' ,filetype=mediaprinttiff', _file)
    _file.close()
    _ftp.quit()

if __name__ == '__main__':
    main()
