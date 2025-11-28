let forecastChart, valChart, cmpChart;

function toastOK(msg){ Swal.fire({icon:'success', title: msg, timer: 1200, showConfirmButton:false}); }
function toastErr(msg){ Swal.fire({icon:'error', title: msg || 'Error', timer: 1500, showConfirmButton:false}); }

async function fetchJSON(url, opts){
  const r = await fetch(url, opts);
  if(!r.ok){ throw new Error(await r.text()); }
  return r.json();
}

function round2(x){ return (x === null || x === undefined) ? null : Math.round(x * 100) / 100; }

function renderChartsAndTable(data){

  const histDates = data.history.dates;
  const histVals  = data.history.values;
  const fDates = data.forecast.rf.dates;  
  const rfVals = data.forecast.rf.values;
  const sxVals = data.forecast.sx.values;

  const ctx1 = document.getElementById('chartForecast').getContext('2d');
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: [...histDates, ...fDates],
      datasets: [
        { label:'History', data: [...histVals, ...Array(fDates.length).fill(null)], borderWidth:2 },
        { label:'RF Forecast', data: [...Array(histVals.length).fill(null), ...rfVals], borderWidth:2 },
        { label:'SARIMAX Forecast', data: [...Array(histVals.length).fill(null), ...sxVals], borderWidth:2 }
      ]
    },
    options: { responsive: true, interaction:{mode:'index',intersect:false} }
  });


  const tb = $('#tblForecast tbody').empty();
  for (let i=0;i<fDates.length;i++){
    tb.append(`<tr><td>${fDates[i]}</td><td>${rfVals[i]}</td><td>${sxVals[i]}</td></tr>`);
  }
  if ($.fn.dataTable.isDataTable('#tblForecast')) {
    $('#tblForecast').DataTable().destroy();
  }
  $('#tblForecast').DataTable();


  const v = data.backtest;
  const ctx2 = document.getElementById('chartVal').getContext('2d');
  if (valChart) valChart.destroy();
  valChart = new Chart(ctx2, {
    type: 'line',
    data: {
      labels: v.dates,
      datasets: [
        { label:'Actual', data: v.y_true, borderWidth:2 },
        { label:'RF (one-step)', data: v.rf_pred, borderWidth:2 },
        { label:'SARIMAX (one-step)', data: v.sx_pred, borderWidth:2 }
      ]
    },
    options: { responsive: true, interaction:{mode:'index',intersect:false} }
  });


  const m = data.backtest.metrics;
  const accRF = (m.RandomForest && m.RandomForest.MAPE != null) ? (100 - m.RandomForest.MAPE * 100) : null;
  const accSX = (m.SARIMAX && m.SARIMAX.MAPE != null) ? (100 - m.SARIMAX.MAPE * 100) : null;

  const ctx3 = document.getElementById('chartCompare').getContext('2d');
  if (cmpChart) cmpChart.destroy();
  cmpChart = new Chart(ctx3, {
    type: 'bar',
    data: {
      labels: ['RandomForest','SARIMAX'],
      datasets: [
        { label:'Accuracy (%) – higher is better', data: [round2(accRF), round2(accSX)] }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          callbacks: {
            label: function(ctx){
              const val = ctx.parsed.y;
              return (val == null) ? 'N/A' : `Accuracy: ${val}%`;
            }
          }
        },
        legend: { display: true }
      },
      scales: {
        y: {
          beginAtZero: true,
          suggestedMax: 100,
          title: { display: true, text: 'Accuracy (%)' }
        }
      }
    }
  });
}

async function loadForecast(){
  const site = $('#siteSelect').val();
  if(!site){ return toastErr('Please choose a site'); }
  Swal.showLoading();
  try{
    const data = await fetchJSON(`/api/forecast?site=${encodeURIComponent(site)}`);
    Swal.close();
    renderChartsAndTable(data);
    toastOK('Forecast loaded');
  }catch(err){
    Swal.close();
    console.error(err);
    toastErr('Failed to load forecast');
  }
}

async function trainAll(){
  Swal.fire({title:'Training all sites...', allowOutsideClick:false, didOpen:()=>Swal.showLoading()});
  try{
    const res = await fetchJSON('/api/train_all', {method:'POST'});
    Swal.close();
    toastOK(`Trained ${res.trained} sites`);
  }catch(err){
    Swal.close();
    console.error(err);
    toastErr('Training failed');
  }
}

function exportCSV(){
  const site = $('#siteSelect').val();
  if(!site){ return toastErr('Please choose a site'); }
  window.location = `/api/export_csv?site=${encodeURIComponent(site)}`;
}

$(document).on('click', '#btnLoad', loadForecast);
$(document).on('click', '#btnTrainAll', trainAll);
$(document).on('click', '#btnExport', exportCSV);
