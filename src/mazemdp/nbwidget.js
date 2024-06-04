class State {
    constructor(container) {
        this.container = container
        let table = document.createElement("table");
        this.cells = []
        for(let i = 0; i < 3; ++i) {
            let row = []
            let tr = document.createElement("tr");
            table.appendChild(tr)
            for(let j = 0; j < 3; ++j) {
                let td = document.createElement("td");
                td.setAttribute("class", "arrow")
                tr.appendChild(td)
                row.push(td)
            }
            this.cells.push(row)
        }

        this.cells[1][1].setAttribute("class", "value")
        this.td_actions = [
            this.cells[0][1],
            this.cells[1][2],
            this.cells[2][1],
            this.cells[1][0]
        ]
        this.td_actions[0].innerHTML = "↑"
        this.td_actions[1].innerHTML = "→"
        this.td_actions[2].innerHTML = "↓"
        this.td_actions[3].innerHTML = "←"

        this.container.appendChild(table)
    }

    set_value(value) {
        // console.log("Setting value", value, Number(value) == value, this.cells)
        if (Number(value) == value) {
            // v value
            this.cells[1][1].innerHTML = value.toFixed(2);
        } else {
            // Q-value
            let q_max = Math.max(...value)
            let q_min = Math.min(...value)
            this.cells[1][1].innerHTML = q_max.toFixed(2);

            for(let a = 0; a < 4; ++a) {
                let alpha = 1
                let color = "0, 255, 0"
                if (value[a] != q_max) {
                    color = "255, 0, 0"
                    alpha = (value[a] - q_min + 1e-8) / (q_max - q_min + 1e-8)
                }
                this.td_actions[a].setAttribute(
                    "style", `color: rgba(${color}, ${alpha})`
                )
            }            
        }
    }
}

function render({ model, el }) {
    console.log("=== creating a new maze ===")
    let steps = [];
    let container = document.createElement("div");

    let div_title = document.createElement("div");
    container.appendChild(div_title)

    let table_maze = document.createElement("table");
    container.appendChild(table_maze)
    table_maze.setAttribute("class", "maze")


    let terminal_states = new Set(model.get('terminal_states'))
    console.log("Terminal states", terminal_states)


    // Cells of the maze
    let cells = [];
    let states = [];

    for(let row of model.get('cells')) {
        let row_divs = []
        let row_div = document.createElement("tr")
        table_maze.appendChild(row_div)

        for(let cell of row) {
            let td = document.createElement("td")
            
            row_divs.push(td)
            row_div.appendChild(td)
            
            if (cell < 0) {
                td.setAttribute("class", "cell wall")
            } else {
                let state = new State(td)
                states.push(state);
                if (terminal_states.has(cell)) {
                    td.setAttribute("class", "cell terminal")
                } else {
                    td.setAttribute("class", "cell")
                }
            }
        }
        cells.push(row_divs)
    }
    

    let div_step = document.createElement("div");
    container.appendChild(div_step)

    div_step.innerHTML = `Step ${model.get("step")} on ${model.get("steps")+1}`;

    model.on("change:step", () => {
        let step = model.get("step");
        console.log(`Step changed: ${step}`)
        if (step < 1 || step > steps.length) {
            return
        }
        let values = steps[step-1]
        for(let s = 0; s < states.length; ++s) {
            states[s].set_value(values[s])
        }

        div_step.innerHTML = `Step ${step} on ${model.get("steps")}`;
    });
    model.on("change:steps", () => {
        div_step.innerHTML = `Step ${model.get("step")} on ${model.get("steps")}`;
    });

    model.on("change:title", () => {
        div_title.innerHTML = `<div>${model.get("title")}</div>`;
    });

    model.on("msg:custom", msg => {
        if (msg.type == "add-step") {
            steps.push(msg.value)
            model.set("steps", steps.length)
            console.log("Added step:", steps.length)
            model.set("step", steps.length-1)
        } else {
            console.log(`new message: ${JSON.stringify(msg)}`);
        }
        console.log("Message processed")
    });

    
    el.appendChild(container);
    model.set("ready", true)
    console.log("Component is ready")
}
export default { render };
