%% Scales and returns the relevant datasets

function [rain_org, rain_org_m1, rain_org_m2, rain_org_v, rain_org_t, ndvi_scaled, ndvi_m, ndvi_v, ndvi_t] = getDatasets()

% Loading data
load proj23.mat ElGeneina

ndvi = ElGeneina.nvdi;
rain_org = ElGeneina.rain_org;

% We rescale the vegetation data to the range -1 to 1
ndvi_scaled = (ndvi-127.5)/127.5;

% Dividing the datasets


% Rain
% [m1, m2, v, t] = [264, 151, 43, 22]

% The rain from the start up to where the validation will start (1 to 415)
rain_org_m1 = rain_org(1:415);
% The rain from the start of the vegetation data up until the validation
rain_org_m2 = rain_org(265:415);
rain_org_v = rain_org(416:458);
rain_org_t = rain_org(459:end);


% Vegetation
% [m1, m2, v, t] = [453, 130, 65]

ndvi_m = ndvi_scaled(1:453); 
ndvi_v = ndvi_scaled(454:583);
ndvi_t = ndvi_scaled(584:end);

