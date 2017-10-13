//
//  database.hpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/10/11.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef database_hpp
#define database_hpp

#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
class database
{
public:
    database();
    ~database();
public:
    std::string dataDir;
    std::string filename;
public:
};

#endif /* database_hpp */
