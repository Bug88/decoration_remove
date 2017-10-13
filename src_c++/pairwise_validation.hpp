//
//  pairwise_validation.hpp
//  pairwise_validation
//
//  Created by willard on 2017/2/14.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef pairwise_validation_hpp
#define pairwise_validation_hpp

#include <stdio.h>

#pragma once

#include <string>
#include <vector>


class PairWiseValidation {
public:
    PairWiseValidation();
    virtual ~PairWiseValidation();
    virtual int IsMatch(const std::vector<char>& buf1, const std::vector<char>& buf2, int* is_match) = 0;
private:
};

class PairWiseValidationFactory {
public:
    static PairWiseValidation* Create(const char* name);
    static int SetGPU(int gpu_id);
};


#endif /* pairwise_validation_hpp */
