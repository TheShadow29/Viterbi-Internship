# cd
s1="/home/agharwal/kitchen/clips/feats/fc7/"
for i in {10..37}
do
    src1="$s1"s"$i"*
	gcp -r $src1 '/media/arka_s/fc23520b-5ce4-4c6e-be90-09d667b2a245/kitchen/clips/feats/fc7/'
	# echo $src1
done
