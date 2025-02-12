import { Component, OnInit } from '@angular/core';
import * as Highcharts from 'highcharts';
import data from '../assets/data.json'; // Import JSON directly

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'RL-DRL-Graphs';
  highcharts = Highcharts;
  chartOptions: any;

  ngOnInit() {
    // Extract categories (states) from data
    const categories = Object.keys(data);  // ["0", "1", "2", ...]

    // Create seriesData based on the updated JSON structure (Q-values and actions)
    const seriesData = Object.entries(data).map(([state, stateData]) => {
      const qValues = stateData.q_values;  // Q-values
      const actions = stateData.actions;   // Actions (Up, Down, Left, Right)

      return {
        name: `State ${state}`,
        data: qValues.map((qValue: number, index: number) => ({
          name: `${actions[index]} (Q=${qValue.toFixed(2)})`,
          y: qValue,
          color: this.getColorForQValue(qValue)  // Optional: Color based on Q-value
        }))
      };
    });

    // Set chart options
    this.chartOptions = {
      chart: {
        type: 'column'  // Using column chart for better visualization of Q-values per action
      },
      title: {
        text: 'Q-table Visualization'
      },
      xAxis: {
        categories: categories,  // Use state names (0, 1, 2, ...)
        title: {
          text: 'States'
        }
      },
      yAxis: {
        title: {
          text: 'Q-value'
        },
        min: -1  // Adjusted for better display of Q-values
      },
      tooltip: {
        pointFormat: 'Q-value: <b>{point.y}</b><br>Action: <b>{point.name}</b>'
      },
      series: seriesData // Series data based on states, actions, and Q-values
    };
  }

  // Helper function to determine color based on Q-value
  getColorForQValue(qValue: number): string {
    if (qValue > 0) {
      return '#28a745'; // Green for positive Q-values
    } else if (qValue < 0) {
      return '#dc3545'; // Red for negative Q-values
    } else {
      return '#ffc107'; // Yellow for zero Q-values
    }
  }
}
