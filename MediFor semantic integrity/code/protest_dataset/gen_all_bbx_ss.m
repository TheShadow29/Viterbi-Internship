img_tdir = '../../data/copied_data/';
f = dir(img_tdir);
f = f(3:end);
bbx = cell(1, length(f));
dict = containers.Map();
for i=1:length(f)
    fname = fullfile(img_tdir, f(i).name);
    bbx{i} = gen_bbx_ss(imread(fname));
    dict(f(i).name) = bbx{i};
    fprintf('Iter %d\n', i)
end
