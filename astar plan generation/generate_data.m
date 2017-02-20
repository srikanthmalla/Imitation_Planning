% input [map, goal, current]
% output [action category]
% 
% action categories
% ------------------------
% [1,0,0,0];%top
% [0,1,0,0]left 
% [0,0,1,0]bottom 
% [0,0,0,1]right 

trials=1000
data=[];input=[];output=[];
for trial=1:trials
    close all;
    [map,start,goal,path,startposind,goalposind,posind]=astardemo
    %we are getting start and goal pos ind interchanged
    if ~isempty(path)
        map=map(:)';
        posind=posind';
        
        %since it is 4 connected
        for i=1:size(posind,2)-1
            currentposind=posind(i);
            input=[input;[map,goalposind,currentposind]];
            if(posind(i+1)-posind(i))==1
                x=[1,0,0,0];%top
            elseif(posind(i+1)-posind(i))==-10
                x=[0,1,0,0];%left
            elseif(posind(i+1)-posind(i))==-1
                x=[0,0,1,0];%bottom
            elseif(posind(i+1)-posind(i))==10
                x=[0,0,0,1];%right
            end
            output=[output;[x]];
        end
    end
end
data=[input, output];
% xlswrite('input1.xls',input)
% xlswrite('output1.xls',output)%actions
% size(data)
% save data
