#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""FTP uploader"""


import os
import ftplib
import ConfigParser


__author__ = "Mikhail Karasikov"
__copyright__ = ""
__email__ = "karasikov@phystech.edu"


def ftp_upload(path, destination, config, ftp=None):
    if ftp is None:
        ftp = ftplib.FTP(config.get('FTPServer', 'host'),
                         config.get('FTPServer', 'user'),
                         config.get('FTPServer', 'passwd'))
        ftp.cwd(config.get('FTPServer', 'path'))
        try:
            ftp.mkd(destination)
        except:
            pass
        ftp.cwd(destination)

    files = os.listdir(path)

    for f in files:
        if os.path.isfile(path + '/' + f):
            fh = open(path + '/' + f, 'rb')
            ftp.storbinary('STOR %s' % f, fh)
            fh.close()
        elif os.path.isdir(path + '/' + f):
            try:
                ftp.mkd(f)
            except:
                pass
            ftp.cwd(f)
            ftp_upload(path + '/' + f, None, None, ftp)
            ftp.cwd('..')
