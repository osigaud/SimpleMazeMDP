import * as React from "react";
import { useMemo } from 'react';

const NORTH = 0;
const SOUTH = 1
const EAST = 2
const WEST = 3


const ShowCell = ({value, policy}: {value: null|number|number[], policy: null|number|number[]}) => {
    let x: null|number = null
    let styles = [{}, {}, {}, {}]

    if (value == null) {
        // do nothing
    } else if (Number(value) == value) {
        // v value
        x = value;
    } else {
        // value is max of Q(s, a)
        x = Math.max(...value)

        if (!policy){
            // Use the Q-value as a policy
            policy = value
        }
    }


    if (policy) {
        if (Number(policy) == policy) {
            // Deterministic
            styles[policy] = {
                color: "rgb(0, 255, 0)"
            }
        } else {
            // Q-value
            let q_max = Math.max(...policy)
            let q_min = Math.min(...policy)
            
            for(let a = 0; a < 4; ++a) {
                let alpha = 1
                let color = "0, 255, 0"
                if (policy[a] != q_max) {
                    color = "255, 0, 0"
                    alpha = (policy[a] - q_min + 1e-8) / (q_max - q_min + 1e-8)
                }
                styles[a] = { "color": `rgba(${color}, ${alpha})` };
            }            
        }
    }

    return <table>
        <tr><td></td><td className="arrow" style={styles[NORTH]}>↑</td><td></td></tr>
        <tr><td className="arrow" style={styles[WEST]}>←</td><td className="value">{x != null ? x?.toFixed(2) : ""}</td><td className="arrow" style={styles[EAST]}>→</td></tr>
        <tr><td></td><td className="arrow" style={styles[SOUTH]}>↓</td><td></td></tr>
    </table>
}

const ShowCells = ({ cells, terminal_states, values, policy, agent_state }) => {
    const terminals = useMemo(() => new Set(terminal_states), [terminal_states]);

    // console.log("Show cells", values, policy, agent_state)
    return <table className="maze">{
        cells.map((row, i) => <tr key={i}>{
            row.map((cell, j) => {
                let s = cells[i][j]
                let cellCN  = "cell"
                if (s < 0) {
                    cellCN = "cell wall"
                } else {
                    if (terminals.has(s)) {
                        cellCN = "cell terminal"
                    }
                }
                // console.log(" cell", i, j, '/', s, " => ", values && values[s], policy && policy[s], agent_state ? agent_state == s : null)
                return <td key={j} className={cellCN} title={`State ${s}`}>{
                    s >= 0 && <ShowCell value={values && values[s]} policy={policy && policy[s]}/>
                }{agent_state == s ? <div class="agent"></div> : null }</td>
            })
        }</tr>)
    }</table>
}



export default function ({ title, terminal_states, cells, values, step, steps, policy, agent_state }) {
    return <div>
        <div>{title}</div>
        <ShowCells cells={cells} terminal_states={terminal_states} policy={policy} values={values} agent_state={agent_state}/>
        <div>Step {step}/{steps}</div>
    </div>
}