# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.

at_biwi = True  # Are you running this code from the ETH Computer Vision Lab (Biwi)?
if os.path.abspath(".").split("/")[1] == "data1":
    project_root = '/data1/su/app/CancerImage/acdc_seg'# /scratch_net/bmicdl03/code/python/acdc_public_segmenter
    data_root = '/data1/su/app/CancerImage/data/ACDC/training'
    test_data_root = '/data1/su/app/CancerImage/data/ACDC/testing'
    local_hostnames = ['localhost']  # used to check if on cluster or not,
elif os.path.abspath(".").split("/")[1] == "home":
    project_root = '/home/yucongl/yuhangl'# /scratch_net/bmicdl03/code/python/acdc_public_segmenter
    data_root = '/home/yucongl/yuhangl/training'
    test_data_root = '/home/yucongl/yuhangl/testing'
    local_hostnames = ['server-pc']  # used to check if on cluster or not,
elif os.path.abspath(".").split("/")[1] == "mnt2":
    project_root = '/mnt2/jinhuas/acdc_seg'# /scratch_net/bmicdl03/code/python/acdc_public_segmenter
    data_root = '/mnt2/jinhuas/acdc_seg/training'
    test_data_root = '/mnt2/jinhuas/acdc_seg/testing'
    local_hostnames = ['server-pc']  # used to check if on cluster or not,
elif os.path.abspath(".").split("/")[1] == "mnt3":
    project_root = '/mnt3/yuhangl/acdc_seg'# /scratch_net/bmicdl03/code/python/acdc_public_segmenter
    data_root = '/mnt3/yuhangl/acdc_seg/training'
    test_data_root = '/mnt3/yuhangl/acdc_seg/testing'
    local_hostnames = ['server-pc']  # used to check if on cluster or not,
else:
    project_root = '/home/wangfeifei/acdc_seg'# /scratch_net/bmicdl03/code/python/acdc_public_segmenter
    data_root = '/home/wangfeifei/ACDC/training'
    test_data_root = '/home/wangfeifei/ACDC/testing'
    local_hostnames = ['1', 'localhost']

# enter the name of your local machine

##################################################################################

log_root = os.path.join(project_root, 'acdc_logdir_4')
preproc_folder = os.path.join(project_root,'preproc_data_4')

def setup_GPU_environment():

    if at_biwi:
        hostname = socket.gethostname()
        print('Running on %s' % hostname)
        if not hostname in local_hostnames:
            logging.info('Setting CUDA_VISIBLE_DEVICES variable...')
            # os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            # logging.info('SGE_GPU is %s' % os.environ['SGE_GPU'])
    else:
        logging.warning('!! No GPU setup defined. Perhaps you need to set CUDA_VISIBLE_DEVICES etc...?')
