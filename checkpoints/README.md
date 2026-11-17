# Checkpoint GraphState (vocab + ma trận kề A)
# File .pkl / .json được tạo bởi teach, train — không commit (xem .gitignore)

# Mặc định:
#   checkpoints/grade_03.pkl   — sau teach-pack + train lớp 3

# Ví dụ:
#   python -m nn_aivie teach-pack lessons/grade_03_pack.yaml
#   python -m nn_aivie train --grade 3 --phase G1 --checkpoint checkpoints/grade_03.pkl -o checkpoints/grade_03.pkl
#   python -m nn_aivie generate --prompt "Em là" --checkpoint checkpoints/grade_03.pkl
