module Gen
  module_function
  def generate(base_name)
    count = 500
    file_name = File.join(File.dirname(__FILE__), base_name)
    file = File.new(file_name, "w")
    file.puts("#{count},5,Positive,PositiveB,PositiveBB,PositiveBBB")
    (1..count).each do
      file.puts gen_number
    end
    file.close
  end

  def gen_number
    arr = []
    (1..5).each do
      ran = (rand(10) + rand).round(1)
      arr.push(ran)
    end
    sum = arr.inject(0.0, :+)
    if sum < 25
      arr.push( sum < 12.5 ? 0 : 1)
    else
      arr.push( sum < 37.5 ? 2 : 3)
    end
    arr.join(",")
  end
end

#main

Gen.generate "train.csv"
Gen.generate "test.csv"