function GetDatasetUSCHAD()

if (length(mfilename()))
    cur_dir = fileparts(which(mfilename()));
else
    cur_dir = pwd;
end

if ~exist([cur_dir, '/USC-HAD.zip'], 'file')
    urlwrite('http://sipi.usc.edu/HAD/USC-HAD.zip', [cur_dir, '/USC-HAD.zip']);
end

if ~length(ls([cur_dir, '/USC-HAD/Subject*']))
    unzip([cur_dir, '/USC-HAD.zip'], cur_dir);
end

end