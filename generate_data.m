trials=100
data=cell(trials,1);
for trial=1:trials
    close all;
    [map,start,goal,path,startposind,goalposind,posind]=astardemo
    if ~isempty(map)
        map=map(:)';
        posind=posind';
        data(trial)={[map,startposind,goalposind,posind]}
    end
end
save data
