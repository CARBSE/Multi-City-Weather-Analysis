$(function(){
  // 1. Typology → Models
  $('#typology').on('change', function(){
    const t = $(this).val();
    const ms = TYP_TO_MODELS[t] || [];
    let opts = `<option value="" disabled selected>-- Select Model --</option>`;
    ms.forEach(m => opts += `<option value="${m}">${m}</option>`);
    $('#comfort_model')
      .html(opts)
      .prop('disabled', ms.length === 0);

    $('#city1, #city2').prop('disabled', true).html(`<option value="" disabled selected>-- Select City --</option>`);
    $('#analyzeBtn').prop('disabled', true);
  });

  // 2. Model → enable states only
  $('#comfort_model').on('change', function(){
    $('#state1, #state2').prop('disabled', false);
    $('#city1, #city2').prop('disabled', true)
      .html('<option value="" disabled selected>-- Select City --</option>');
    $('#analyzeBtn').prop('disabled', true);
  });

  // 3. State→City cascade
  function bindStateToCity(stateSel, citySel) {
    $(stateSel).on('change', function(){
      const st = $(this).val();
      $(citySel)
        .prop('disabled', true)
        .html('<option>Loading…</option>');
      $.getJSON('/api/cities', { state: st }, function(data){
        let o = '<option value="" disabled selected>-- Select City --</option>';
        data.cities.forEach(c => o += `<option value="${c}">${c}</option>`);
        $(citySel).html(o).prop('disabled', false);
      });
      $('#analyzeBtn').prop('disabled', true);
    });
  }
  bindStateToCity('#state1', '#city1');
  bindStateToCity('#state2', '#city2');

  // 4. enable analyze when both cities chosen
  $('#city1, #city2').on('change', function(){
    const ready = $('#city1').val() && $('#city2').val();
    $('#analyzeBtn').prop('disabled', !ready);
  });

  // 5. Analyze → map, stats, charts
  $('#cityForm').on('submit', function(e){
    e.preventDefault();
    const typ   = $('#typology').val();
    const model = $('#comfort_model').val();
    const c1    = $('#city1').val();
    const c2    = $('#city2').val();
    const ts    = Date.now();

    $('#headerCities').text(`${c1} vs ${c2}`);
    $('#headerTyp').text(model);

    $('#mapMulti').attr('src',
      `/map_preview?city1=${encodeURIComponent(c1)}&city2=${encodeURIComponent(c2)}&ts=${ts}`
    );
    $('#profile1').attr('src',
      `/static/images/city-profile-for-single-city/${encodeURIComponent(c1)}.jpg?ts=${ts}`
    );
    $('#profile2').attr('src',
      `/static/images/city-profile-for-single-city/${encodeURIComponent(c2)}.jpg?ts=${ts}`
    );

    $('#label1').text(c1);
    $('#stats1-heading').text(`Stats for ${c1}`);
    $('#label2').text(c2);
    $('#stats2-heading').text(`Stats for ${c2}`);

    $.post('/analyze', { typology: typ, model: model, city1: c1, city2: c2 }, function(res){
      $('#table1').html(res.table1);
      $('#table2').html(res.table2);
      $('#dlStats1').attr('href',
        `/download/stats1?city1=${encodeURIComponent(c1)}&model=${encodeURIComponent(model)}&ts=${ts}`
      );
      $('#dlStats2').attr('href',
        `/download/stats2?city2=${encodeURIComponent(c2)}&model=${encodeURIComponent(model)}&ts=${ts}`
      );
    });


// After header, map, profiles, stats are set…
$.when(
  $.getJSON('/api/minmax', { city: c1, model: model }),
  $.getJSON('/api/minmax', { city: c2, model: model })
).done(function(r1, r2){
  const mn = Math.min(r1[0].min_y, r2[0].min_y);
  const mx = Math.max(r1[0].max_y, r2[0].max_y);

  console.log("Global caps:", mn, mx); // verify in console

  $('#chart1').attr('src',
    `/chart/comfort1?city1=${encodeURIComponent(c1)}&model=${encodeURIComponent(model)}&min_y=${mn}&max_y=${mx}&ts=${Date.now()}`
  );
  $('#chart2').attr('src',
    `/chart/comfort2?city2=${encodeURIComponent(c2)}&model=${encodeURIComponent(model)}&min_y=${mn}&max_y=${mx}&ts=${Date.now()}`
  );
});

    $('#comfort-legend').attr('src', `/chart/legend?ts=${ts}`);
    $('#dlBothBands').off('click').on('click', () => {
      const url = `/chart/combined_comfort?chart1=${encodeURIComponent(`/chart/comfort1?city1=${c1}&model=${model}`)}&chart2=${encodeURIComponent(`/chart/comfort2?city2=${c2}&model=${model}`)}&ts=${ts}`;
      window.open(url, '_blank');
    });

    $('#chartHours').attr('src',
      `/chart/comfort_hours?city1=${encodeURIComponent(c1)}&city2=${encodeURIComponent(c2)}&model=${encodeURIComponent(model)}&ts=${ts}`
    );
    $('#dlChartHours').attr('href',
      `/download/chart_hours?city1=${encodeURIComponent(c1)}&city2=${encodeURIComponent(c2)}&model=${encodeURIComponent(model)}&ts=${ts}`
    );

    $('#analysisSection').removeClass('d-none');
  });
});
