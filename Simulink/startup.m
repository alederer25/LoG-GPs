% Copyright (c) by Alejandro Ordonez-Conejo (TUM) under BSD License 
% Last modified: Armin Lederer 2022-02
addpath(genpath('source code'));
addpath(genpath('library'));

open('library/RealTimeLearning.slx')
set_param('RealTimeLearning','lock','off');
set_param(gcs,'EnableLBRepository','on');
save_system;
set_param('RealTimeLearning','lock','on');
close_system('RealTimeLearning');