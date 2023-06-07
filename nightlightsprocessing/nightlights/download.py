#!/usr/bin/env python3

# Taken and modified from:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#python

# Attempts to do HTTP Gets with urllib2(py2) urllib.requets(py3) or subprocess
# if tlsv1.1+ isn't supported by the python ssl module
#
# Will download csv or json depending on which python module is available
#
# IMPORTANT: This script should be used with a destination e.g:
# https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A1/2014/305

from __future__ import (division, print_function, absolute_import, unicode_literals)

import argparse
import os
import os.path
import shutil
import sys
from urllib.request import urlopen, Request, URLError, HTTPError
# import chardet

from io import StringIO

################################################################################

# you will need to replace the following line with the location of a
# python web client library that can make HTTPS requests to an IP address.
USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')

# this is the choice of last resort, when other attempts have failed
def getcURL(url, headers=None, out=None):
    import subprocess
    try:
        print('trying cURL', file=sys.stderr)
        # --fail: Specifies that cURL should fail if the HTTP response indicates an error.
        # -sS: Enables silent mode and shows errors if they occur.
        # -L: Instructs cURL to follow redirects if the server responds with a redirect.
        # -b session: Uses the cookie data from the file named session.
        # --get: Specifies that you want to perform an HTTP GET request.
        args = ['curl', '--fail', '-sS', '-L', '-b session', '--get', url]
        for (k,v) in headers.items():
            args.extend(['-H', ': '.join([k, v])])
        if out is None:
            # python3's subprocess.check_output returns stdout as a byte string
            result = subprocess.run(args, check=True, capture_output=True)
            # result2 = subprocess.check_output(args)
            is_bytes_instance = isinstance(result, bytes)
            print('is_bytes_instance', is_bytes_instance)

            if is_bytes_instance:
                return result.decode('utf-8')
            else:
                return result
        else:
            subprocess.call(args, stdout=out)
    except subprocess.CalledProcessError as e:
        print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
    return None
    
# read the specified URL and output to a file
def geturl(url, token=None, out=None):
    headers = { 'user-agent' : USERAGENT }

    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
        print('headers ----------', headers)
    try:
        import ssl
        from urllib.request import urlopen, Request, URLError, HTTPError

        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        # CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            print('URL ----------', url)
            request = Request(url, headers=headers)
            print('request ----------', request)

            fh = urlopen(request, context=CTX)
            if out is None:
                return fh.read().decode('utf-8')
            else:
                shutil.copyfileobj(fh, out)
        except HTTPError as e:
            print('TLSv1_2 : HTTP GET error code: %d' % e.code, file=sys.stderr)
            getcURL(url, headers, out)
        except URLError as e:
            print('TLSv1_2 : Failed to make request: %s' % e.reason, file=sys.stderr)
            return getcURL(url, headers, out)

    except AttributeError:
      return getcURL(url, headers, out)


################################################################################


DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"


def sync(src, destination, token):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = {}
        #  Reads a .csv file which represents all the data from the url given
        files['content'] = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, token)), skipinitialspace=True) ]

    except ImportError:
        print('ERROR')
        import json
        files = json.loads(geturl(src + '.json', token))
    
     
    print('FILES CONTENT', len(files['content']))
    # next(iter(my_dict))

    # use os.path since python 2/3 both support it while pathlib is 3.4+
    ONLY_FIRST_FILE_FOR_NOW = 2
    for f in files['content'][:ONLY_FIRST_FILE_FOR_NOW]:
        print('FILE', f)
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path = os.path.join(destination, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:                 # size FROM RESPONSE
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, token)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path) or os.path.getsize(path) == 0:    # filesize FROM OS
                    print('\ndownloading: ' , path)
                    with open(path, 'w+b') as fh:
                        geturl(url, token, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
    return 0

def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument('-s', '--source', dest='source', metavar='URL', help='Recursively download files at URL', required=True)
    parser.add_argument('-d', '--destination', dest='destination', metavar='DIR', help='Store directory structure in DIR', required=True)
    parser.add_argument('-t', '--token', dest='token', metavar='TOK', help='Use app token TOK to authenticate', required=True)
    args = parser.parse_args(argv[1:])
    # If the directory doesn't exist, create the path
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
    return sync(args.source, args.destination, args.token)

if __name__ == '__main__':
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
        