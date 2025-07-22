//
// Created by Song.H.L on 2025/7/5.
//

#ifndef SRC_READ_OPLINSTANCE_H
#define SRC_READ_OPLINSTANCE_H

# include "instance.h"

/*
 * 数据来源
 * @article{ponza2016optimization,
        title={Optimization of drone-assisted parcel delivery},
        author={Ponza, Andrea},
        year={2016}
    }
 */

Instance readPonzaInstance(const string &path);

#endif //SRC_READ_OPLINSTANCE_H
