from ftplib import FTP
import fnmatch
import shutil
import os
import traceback
from RLA.easy_log import logger

import pysftp

class FTPHandler(object):

    def __init__(self, ftp_server, username, password, ignore=None):
        self.ftp_server = ftp_server
        self.username = username
        self.password = password
        self.ftp = self.ftpconnect()
        logger.info("login success.")
        self.ignore = ignore
        self.ignore_rules = []
        if self.ignore is not None:
            self.__init_gitignore()
        assert isinstance(self.ftp, FTP)

    def __init_gitignore(self):

        with open(self.ignore, 'r') as f:
            for line in f.readlines():
                if line[0] == '#':
                    continue
                elif line[0] == '\n':
                    continue
                else:
                    if line[-1] == '\n':
                        line = line[:-1]
                    self.ignore_rules.append(line)

    def ignore_match(self, files):
        filenames = files
        for ignore in self.ignore_rules:
            filenames = [n for n in filenames if not fnmatch.fnmatch(n, ignore)]
        return filenames

    def ftpconnect(self):
        ftp = FTP()
        ftp.set_debuglevel(0)
        ftp.connect(self.ftp_server, 21, timeout=60)
        ftp.login(self.username, self.password)
        logger.warn("login succeed")
        return ftp

    def all_file_search(self, root_path, files, filter_length):
        all_files = self.ftp.nlst(root_path)
        assert all_files is not []
        if len(all_files) == 1:
            try:
                assert self.ftp.size(all_files[0]) is not None
                files.append(all_files[0][filter_length:])
                return
            except Exception as e:
                logger.warn("WARNING in all file {}".format(all_files))
                logger.warn(traceback.format_exc())

        for f in all_files:
            self.all_file_search(f, files, filter_length)

    def upload_file(self, remote_dir, local_dir, local_file):
        self.ftp = self.ftpconnect()
        self.ftp.timeout = 600
        bufsize = 1024
        with open(os.path.join(local_dir, local_file), 'rb') as fp:
            try:
                self.ftp.cwd(remote_dir)
            except Exception as e:
                # directory doesn't not exists. create it.
                dirpath = remote_dir.replace('\\', '/')
                tmp = dirpath.split('/')
                dirs = []
                for _ in tmp:
                    if len(dirs) == 0:
                        dirs.append(_)
                        continue
                    dirs.append(dirs[-1] + '/' + _)
                success = False
                expection = Exception
                for _ in dirs:
                    try:
                        self.ftp.mkd(_)
                        success = True
                    except Exception as e:
                        expection = e
                        e_str = str(e)
                        if '550' in e_str and 'File exists' in e_str:
                            continue
                if not success:
                    raise expection
                logger.warn('create dir succeed {}'.format(remote_dir))
                self.ftp.cwd(remote_dir)
            self.ftp.storbinary('STOR ' + local_file, fp, bufsize)
        self.close()

    def download_file(self, remote_file, local_file):
        bufsize = 1024
        logger.info("try download {}".format(local_file))
        if not os.path.isfile(local_file):
            fp = open(local_file, 'wb')
            logger.info("new file {}".format(local_file))
            self.ftp.retrbinary('RETR ' + remote_file, fp.write, bufsize)
        elif self.ftp.size(remote_file) != os.path.getsize(local_file):
            fp = open(local_file, 'wb')
            logger.info("update file {}".format(local_file))
            self.ftp.retrbinary('RETR ' + remote_file, fp.write, bufsize)
        else:
            logger.info("skip download file {}".format(remote_file))
    def get_dir(self, path):
        split_path = path.split('/')
        return '/'.join(split_path[:-1])

    def download_files(self, files, remote_root, local_root):
        for file in files:
            remote_path = remote_root + file
            local_path = local_root + file
            dir = self.get_dir(local_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.download_file(remote_path, local_path)

    def close(self):
        self.ftp.quit()
        self.ftp.close()

class SFTPHandler(FTPHandler):

    def __init__(self, sftp_server, username, password, ignore=None):
        self.sftp_server = sftp_server
        self.username = username
        self.password = password
        self.sftp = self.sftpconnect()
        logger.info("login success.")
        self.ignore = ignore
        self.ignore_rules = []
        if self.ignore is not None:
            self.__init_gitignore()

    def sftpconnect(self):
        sftp = pysftp.Connection(self.sftp_server, username=self.username, password=self.password)
        logger.warn("login succeed")
        return sftp

    def all_file_search(self, root_path, files, filter_length):
        if root_path[-1] != '/':
            root_path += '/'
        all_files = [root_path + x for x in self.sftp.listdir(root_path)]
        assert all_files is not []
        if len(all_files) == 1:
            try:
                assert self.sftp.stat(all_files[0]).st_size is not None
                files.append(all_files[0][filter_length:])
                return
            except Exception as e:
                logger.warn("WARNING in all file {}".format(all_files))
                logger.warn(traceback.format_exc())

        for f in all_files:
            if self.sftp.isdir(f):
                self.all_file_search(f, files, filter_length)

    def upload_file(self, remote_dir, local_dir, local_file):
        self.sftp = self.sftpconnect()
        try:
            self.sftp.cwd(remote_dir)
        except Exception as e:
            # directory doesn't not exists. create it.
            dirpath = remote_dir.replace('\\', '/')
            tmp = dirpath.split('/')
            dirs = []
            for _ in tmp:
                if len(dirs) == 0:
                    dirs.append(_)
                    continue
                dirs.append(dirs[-1] + '/' + _)
            success = False
            expection = Exception
            for _ in dirs:
                try:
                    self.sftp.mkdir(_)
                    success = True
                except Exception as e:
                    expection = e
                    e_str = str(e)
                    if '550' in e_str and 'File exists' in e_str:
                        continue
            if not success:
                raise expection
            logger.warn('create dir succeed {}'.format(remote_dir))
            self.sftp.cwd(remote_dir)
        self.sftp.put(local_dir + local_file)
        self.close()

    def download_file(self, remote_file, local_file):
        self.sftp = self.sftpconnect()
        logger.info("try download {}".format(local_file))
        if not os.path.isfile(local_file):
            logger.info("new file {}".format(local_file))
            self.sftp.get(remote_file, local_file)
        elif self.sftp.stat(remote_file).st_size != os.path.getsize(local_file):
            logger.info("update file {}".format(local_file))
            self.sftp.get(remote_file, local_file)
        else:
            logger.info("skip download file {}".format(remote_file))

    def close(self):
        self.sftp.close()
