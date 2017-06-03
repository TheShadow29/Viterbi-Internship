# $fold='/arka_data/NC2017_Dev1_Beta4'
# $fold=/home/arka_s/internship_files/Viterbi-Internship
for f in *.jpg ; do
  if [[ $(file -b --mime-type "$f") = image/png ]] ; then
    mv "$f" "${f/%.jpg/.png}"
  fi
done
