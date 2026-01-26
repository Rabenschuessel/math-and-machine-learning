
sed -i -e "s/experiment-//g" -e 's/-epochs//g' -e 's/-pretrained//g' -e 's/\(gm\|pz\)-\([0-9]*\)/\1_\2/g' $@
