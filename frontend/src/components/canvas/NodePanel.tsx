import { useState } from "react";
import { NODE_CATEGORIES, getAllNodes, NodeMeta } from "./nodeRegistry";
import { useCanvasStore } from "@/store/canvasStore";

export function NodePanel() {
  const [search, setSearch] = useState("");
  const addNode = useCanvasStore((s) => s.addNode);

  const handleDragStart = (e: React.DragEvent, meta: NodeMeta) => {
    e.dataTransfer.setData("application/node-meta", JSON.stringify(meta));
    e.dataTransfer.effectAllowed = "move";
  };

  const allNodes = getAllNodes();
  
  const filteredCategories = NODE_CATEGORIES.reduce<
    Record<string, NodeMeta[]>
  >((acc, category) => {
    const nodesInCategory = allNodes.filter(
      (n) => n.category === category.id
    );
    
    const filtered = nodesInCategory.filter(
      (n) =>
        n.label.toLowerCase().includes(search.toLowerCase()) ||
        (n.description && n.description.toLowerCase().includes(search.toLowerCase()))
    );
    
    if (filtered.length > 0) acc[category.label] = filtered;
    return acc;
  }, {});

  return (
    <div
      style={{
        width: 200,
        height: "100%",
        background: "#0f172a",
        borderRight: "1px solid #1e293b",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* 搜索框 */}
      <div style={{ padding: "10px 8px", borderBottom: "1px solid #1e293b" }}>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索节点..."
          style={{
            width: "100%",
            background: "#1e293b",
            border: "1px solid #334155",
            borderRadius: 6,
            color: "#e2e8f0",
            padding: "6px 8px",
            fontSize: 12,
            outline: "none",
            boxSizing: "border-box",
          }}
        />
      </div>

      {/* 节点列表 */}
      <div style={{ flex: 1, padding: "8px 0" }}>
        {Object.entries(filteredCategories).map(([category, nodes]) => (
          <div key={category}>
            <div
              style={{
                padding: "4px 12px",
                color: "#475569",
                fontSize: 10,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
              }}
            >
              {category}
            </div>
            {nodes.map((meta) => (
              <div
                key={meta.op_type}
                draggable
                onDragStart={(e) => handleDragStart(e, meta)}
                title={meta.description}
                style={{
                  margin: "2px 8px",
                  padding: "6px 10px",
                  borderRadius: 6,
                  background: "#1e293b",
                  cursor: "grab",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  transition: "background 0.15s",
                }}
                onMouseEnter={(e) =>
                  ((e.currentTarget as HTMLDivElement).style.background = "#334155")
                }
                onMouseLeave={(e) =>
                  ((e.currentTarget as HTMLDivElement).style.background = "#1e293b")
                }
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: meta.color,
                    flexShrink: 0,
                  }}
                />
                <span style={{ color: "#cbd5e1", fontSize: 12 }}>
                  {meta.label}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}