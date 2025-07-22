#ifndef SRC_READ_MURRYINSTANCE_H
#define SRC_READ_MURRYINSTANCE_H

# include "instance.h"

/*
 * 数据来源
 * @article{murray2015flying,
      title={The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery},
      author={Murray, Chase C and Chu, Amanda G},
      journal={Transportation Research Part C: Emerging Technologies},
      volume={54},
      pages={86--109},
      year={2015},
      publisher={Elsevier}
    }
 */

Instance readMurrayInstance(const string &path);

#endif //SRC_READ_MURRYINSTANCE_H
