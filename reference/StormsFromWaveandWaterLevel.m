% Code to find storms (duration > 8hr with Hs > 2m) at Masonboro Inlet Buoy and
% Wrightville tide gauge
% Input: YEAR (line 17)
% Output: output_table with these fields: Start date, end date, Max Hs, Duration
% in hours, Max Water Level and Storm surge

% Author: Liliana Velasquez Montoya
% Last edited: 2017/08/05 
% edited to be able to pull from Duck 17m buoy and Duck tide gauge.
% edited 2017/03/21 to fix max WL and max surge so they do not need to
% occur at the same time
% Edited 10/23/21 ejs so that it is pulling only from the Duck buoy and
% gauge
% Edited 5/11/26 Liz Farquhar so that it is pulling only from the Masonboro buoy and
% Wrightsville gauge


clc
clear all

% Read data from the NDBC website for a given year
YEAR='2026';

% User input
messg=strcat('THE YEAR YOU WILL PROCESS IS: ', YEAR);
disp(messg);
%Set to Duck Pier
    %StationWaves='41110'; % Masonboro Inlet 50 ft depth wave buoy
    StationWaves='44056'; %Duck FRF
    %StationWL='8658163'; % Wrightsville Pier
    StationWL='8651370'; %Duck Pier
    filename=strcat('StormTable_Duck50ft_',YEAR,'.xlsx');


options = weboptions('Timeout',60);
webaddressGauge=strcat('http://www.ndbc.noaa.gov/view_text_file.php?filename=',StationWaves,'h',YEAR,'.txt.gz&dir=data/historical/stdmet/');                              
data = webread(webaddressGauge);
%data = fileread('DuckWave2026JantoApr.txt');
data1=strsplit(data,'\n'); % split data into rows

% Extract times (t) and significant wave height (m)
hs=zeros(length(data1)-3,1);
j=1;
for i=3:length(data1)-1
    data2=strsplit(data1{i},' '); % split data into columns
    if str2double(data2{9}) < 50 
        hs(j,1)= str2double(data2{9});
        t(j,1)=datetime([str2num(data2{1}),str2num(data2{2}),str2num(data2{3}),str2num(data2{4}),str2num(data2{5}),0]);
        j=j+1;
    end
end

% Find the location of the different events with Hs>2m and durations > 8 hr
j=1;
greaterthan2m=find(hs>1.5);
initial=t(greaterthan2m(1));
eight_hours=duration([8,0,0]);
l=(greaterthan2m(1));
for k=1:length(greaterthan2m)-1
    if greaterthan2m(k+1)-greaterthan2m(k)>1
        final=t(greaterthan2m(k));
        Duration=final-initial;
        if Duration > eight_hours
            times(j,1)=initial; 
            times(j,2)=final;
            dur(j,1)=Duration;
            maxHs(j,1)=max(hs(l:greaterthan2m(k),1));
            j=j+1;
            l=greaterthan2m(k+1);
            event(j)=k;
        end
        initial=t(greaterthan2m(k+1));       
    end
end

% Convert durations to decimal hours
dur_vector=datevec(dur,'HH:MM:SS');
dur_decimal= dur_vector(:,1)*8760+dur_vector(:,2)*730+dur_vector(:,3)*24+dur_vector(:,4)+dur_vector(:,5)./60+dur_vector(:,6)/3600;

% Retrieve WL 
maxWL=zeros(length(times(1,2)),1);
surge=zeros(length(times(1,2)),1);
for m=1:length(times)
    % initial date
    y1=num2str(year(times(m,1)));
    m1=num2str(month(times(m,1)));
    if length(m1)<2
        m1=strcat('0',m1);
    end
    d1=num2str(day(times(m,1)));
    if length(d1)<2
        d1=strcat('0',d1);
    end
    inidate=strcat(y1,m1,d1);
    % final date
    y2=num2str(year(times(m,2)));
    m2=num2str(month(times(m,2)));
    if length(m2)<2
        m2=strcat('0',m2);
    end
    d2=num2str(day(times(m,2)));
    if length(d2)<2
        d2=strcat('0',d2);
    end
    findate=strcat(y2,m2,d2);

    WLObsGauge=webread(strcat('https://tidesandcurrents.noaa.gov/api/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date=',inidate,'&end_date=',findate,'&datum=NAVD&station=',StationWL,'&time_zone=GMT&units=metric&format=csv'));
    %WLObsGauge=webread(strcat('https://tidesandcurrents.noaa.gov/api/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date=20260101&end_date=20260430&datum=NAVD&station=',StationWL,'&time_zone=GMT&units=metric&format=csv'));
    WLPredictGauge=webread(strcat('https://tidesandcurrents.noaa.gov/api/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date=',inidate,'&end_date=',findate,'&datum=NAVD&station=',StationWL,'&time_zone=GMT&units=metric&interval=h&format=csv'));
    %WLPredictGauge=webread(strcat('https://tidesandcurrents.noaa.gov/api/datagetter?product=predictions&application=NOS.COOPS.TAC.WL&begin_date=20260101&end_date=20260430&datum=NAVD&station=',StationWL,'&time_zone=GMT&units=metric&interval=h&format=csv'));

    % Find the initial and final dates closest to the storms 
    diff_ini=datenum(abs(WLObsGauge.DateTime-times(m,1)));
    diff_fin=datenum(abs(WLObsGauge.DateTime-times(m,2)));
    
    minddiffini=min(diff_ini);
    minddifffin=min(diff_fin);
    
    locini=find(diff_ini(:)==minddiffini,1,'last');
    locfin=find(diff_fin(:)==minddifffin,1,'last');
    
    allsurges=WLObsGauge.WaterLevel(locini:locfin)-WLPredictGauge.Prediction(locini:locfin);
    surge(m,1)=max(allsurges);
    maxWL(m,1)=max(WLObsGauge.WaterLevel(locini:locfin));
         
end

% Generate output table with the events with Hs>2m with duration>8hrs
% including the following columns: Start date, end date, Max Hs, Duration
% in hours, Max Water Level and Storm surge

StartTime=datestr(times(:,1));
EndTime=datestr(times(:,2));
MaxHs_ft=maxHs(:,1)./0.3048;
Duration_hr=dur_decimal;
MaxWaterLevel_ftNAVD=maxWL(:,1)./0.3048;
Surge_ft=surge(:,1)./0.3048;

output_table=table(StartTime,EndTime,MaxHs_ft,Duration_hr,MaxWaterLevel_ftNAVD,Surge_ft);
outtext=strcat('Find the storm data in output_table and in the file: ',filename);
disp(outtext);
outtable=table2cell(output_table);
heading={'StartDate','EndDate','MaxHs_ft','Duration_Hr','MaxWL_ftNAVD','Surge_ft'};
xlswrite(filename,heading,'storms','A1')
xlswrite(filename,outtable,'storms','A2')

clearvars -except output_table


