#  ---------------------------------------------------------------
#  Copyright (c) 2025 Bruno B. Englert. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------

import os


def mkdir(dir_name):
    assert type(dir_name), str
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
