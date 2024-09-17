import * as React from "react";
import { useMemo } from 'react';

/**
 * 
 * @param cells A of states
 * @returns 
 */
const getStatesMap = (cells: number[][]) => {
    // A matrix of cells
    const cell2state: (null|number)[][] = []
    let current_state = 0

    cells.forEach(
        (row) => {
            let row_map: (null|number)[] = []
            row.forEach(
                (cell) => {
                    if (cell >= 0) {
                        row_map.push(current_state)
                        current_state += 1
                    } else {
                        row_map.push(null)
                    }
                }
            )
            cell2state.push(row_map)
        }
    )
    return cell2state
}

const NORTH = 0;
const SOUTH = 1
const EAST = 2
const WEST = 3


const ShowCell = ({value}) => {
    let x: null|number = null
    let styles = [{}, {}, {}, {}]
    if (value == null) {
        // do nothing
    } else if (Number(value) == value) {
        // v value
        x = value;
    } else {
        // Q-value
        let q_max = Math.max(...value)
        let q_min = Math.min(...value)
        x = q_max

        for(let a = 0; a < 4; ++a) {
            let alpha = 1
            let color = "0, 255, 0"
            if (value[a] != q_max) {
                color = "255, 0, 0"
                alpha = (value[a] - q_min + 1e-8) / (q_max - q_min + 1e-8)
            }
            styles[a] = { "color": `rgba(${color}, ${alpha})` };
        }            
    }

    return <table>
        <tr><td></td><td className="arrow" style={styles[NORTH]}>↑</td><td></td></tr>
        <tr><td className="arrow" style={styles[WEST]}>←</td><td className="value">{x != null ? x?.toFixed(2) : ""}</td><td className="arrow" style={styles[EAST]}>→</td></tr>
        <tr><td></td><td className="arrow" style={styles[SOUTH]}>↓</td><td></td></tr>
    </table>
}

const ShowCells = ({ cells, terminal_states, values }) => {
    const terminals = useMemo(() => new Set(terminal_states), [terminal_states]);
    const cell2state = useMemo(() => getStatesMap(cells), [cells]);

    return <table className="maze">{
        cells.map((row, i) => <tr key={i}>{
            row.map((cell, j) => {
                let s = cell2state[j][i]
                let cellCN  = "cell"
                if (s === null) {
                    cellCN = "cell wall"
                } else {
                    if (terminals.has(s)) {
                        cellCN = "cell terminal"
                    }
                }
                return <td key={j} className={cellCN} title={`State ${s}`}>{
                    s !== null && <ShowCell value={values && values[s]}/>
                }</td>
            })
        }</tr>)
    }</table>
}



export default function ({ title, terminal_states, cells, values, step, steps }) {
    return <div>
        <div>{title}</div>
        <ShowCells cells={cells} terminal_states={terminal_states} values={values}/>
        <div>Step {step}/{steps}</div>
    </div>
}