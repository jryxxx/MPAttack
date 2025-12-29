import bpy # type: ignore
import bmesh # type: ignore

# 输出路径（自己改）
output_path = "/your/path/selected_face_ids.txt"

obj = bpy.context.object
mesh = obj.data

# 获取当前编辑 mesh
bm = bmesh.from_edit_mesh(mesh)

# 收集选中 faces 的 ID
selected_ids = [f.index for f in bm.faces if f.select]

# 写入 txt 文件
with open(output_path, "w") as f:
    for face_id in selected_ids:
        f.write(f"{face_id}\n")

print(f"导出完成！共导出 {len(selected_ids)} 个面 ID")
print("保存到:", output_path)
