% Copyright (c) by Alejandro Ordonez-Conejo (TUM) under BSD License 
% Last modified: Armin Lederer 2022-02
function blkStruct = slblocks
        v = version('-release');
        if(strcmp(v,'2019a'))
            Browser.Library = 'RealTimeLearning';
            Browser.Name = 'Real Time Learning';
            
            blkStruct.Browser = Browser;
        end